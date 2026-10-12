// The EmbeddingGemma 2 `.litertlm` embedding pass over LiteRT-LM's
// EmbeddingEngine C API — what `LiteRtEmbeddingBackend.describe` hands core for
// a self-contained `.litertlm` embedder spec. Core builds it inside its
// embedding worker isolate, hosts it and closes it (R5 §4).
//
// The bundle carries its own tokenizer, so nothing here touches core's
// tokenizer registry: the pass hands the engine `taskType.prefix + text` and
// the engine tokenizes it, adding BOS and EOS (special tokens pinned on, D26).
//
// What it decides, and why (R5 D21-D26, §4.8, §4.10, measured in
// eg2_design_backup/measurements/REPORT.md):
// - Capabilities come from ModelInfo, read and freed BEFORE any settings or
//   engine exist: model type, dimension, text signatures, and image support
//   from `vision_signature_selection` — never `supports_input_modality`, which
//   claims vision and audio on the text-only 270m (D23). A declared medium is
//   refused here, so no engine (and no tower cache) is made for a request that
//   cannot be served. Text, and images on a bundle with a vision encoder (the
//   440m); audio arrives in 1.16.0.
// - Images: one encoded file per request, decoded by the engine (stb: JPEG,
//   PNG, GIF, BMP — reported at load, so core refuses anything else before
//   sending), at 140 vision tokens per image or the opt-in 70 (D25). Tower
//   backends stay NULL: the vision tower runs on the main backend (D26).
// - The input cap is set explicitly to 1024 tokens (D21): the bundles' metadata
//   says 1024 but no C getter exposes it, and one 8192-token input peaked at
//   +10.25 GB. `maxInputTokens` is the smallest text signature at or above the
//   setting — the limit the engine enforces.
// - The engine always gets an existing, writable cache directory that this
//   pass owns (D22), under the platform cache directory (never backed up):
//   without one it runs with no weight cache (+290 MB) or writes
//   `.xnnpack_cache` files next to the model. Our part of the path is kept
//   short — `flutter_edge_ai/emb/<8-hex hash>-<base36 size>-<base36 mtime>`
//   (see `liteRtLmCacheDirName`: Windows MAX_PATH), and a load that leaves it empty is reported. Caches of earlier
//   copies of the same file name are deleted only after the engine exists,
//   and not while they were touched in the last 10 minutes.
// - Overflow (§4.8): options use kError; a NULL response with status 3
//   (InvalidArgument) on the single, well-formed text part this pass built is
//   an overflow — the only InvalidArgument reachable for such a request at
//   v0.18.0. The count is not parsed out of the message (owner decision), so
//   `actualTokens` is null until the response getters land (P3). On an image
//   request code 3 is a decode failure (a corrupt or truncated file; an image
//   cannot reach the 1024-token cap at 140 tokens), so it stays an engine
//   error.
// - Ownership (§4.10): every create has its delete on every path, and the
//   native error (thread-local) is copied in the same synchronous segment as
//   the failing call, before any other native call.
// - NPU (an EmbeddingGemma 2 per-SoC bundle, placed by `describe()`, see
//   litert_lm_npu.dart): the main backend is "npu", the tower backends stay
//   NULL so present towers run on it too (D26), and the Qualcomm dispatch
//   directory is handed to every executor (main, vision, audio; upstream
//   ignores a tower the bundle lacks). `engine_create` succeeds even when the
//   NPU graph cannot run (measured: it only logs a failed warm-up), so the load
//   embeds one probe text and refuses unless it gets a finite, non-zero
//   vector of the right size. On Linux every NPU call runs with SIGPROF
//   blocked: the debug VM profiler's signal breaks QNN (see sigprof_mask.dart),
//   and this pass calls the engine on its own thread, synchronously.

import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:math' as math;

import 'package:crypto/crypto.dart' show sha256;
import 'package:ffi/ffi.dart';
import 'package:flutter/foundation.dart' show visibleForTesting;
import 'package:flutter_edge_ai/engine.dart';

import '../ffi/litert_lm_embedding_bindings.dart';
import '../ffi/litert_lm_library.dart';
import '../ffi/sigprof_mask.dart';
import 'embedding_parts.dart';
import 'litert_lm_embedding_common.dart';
import 'litertlm_npu_target.dart';
import 'vision_budget.dart';

// Shared with the web pass; re-exported so this file's users keep one import.
export 'litert_lm_embedding_common.dart'
    show
        liteRtLmEngineTag,
        liteRtLmImageFormats,
        liteRtLmMaxInputLength,
        liteRtLmNoAudioRefusal;

/// `kLiteRtLmStatusOk` / `kLiteRtLmStatusInvalidArgument` (`c/error_reporter.h`).
const _statusOk = 0;
const _statusInvalidArgument = 3;

/// What `LiteRtEmbeddingBackend.describe` hands core for a `.litertlm`
/// embedder. Crosses to core's worker isolate: strings, a set of enums and an
/// int, all sendable.
final class LiteRtLmEmbeddingConfig {
  const LiteRtLmEmbeddingConfig({
    required this.modelPath,
    required this.cacheBase,
    this.declaredMedia = const {},
    this.numThreads,
    this.visionTokensPerImage = liteRtLmDefaultVisionTokensPerImage,
    this.npuTarget,
    this.npuDispatchDir,
  }) : assert(
         (npuTarget == null) == (npuDispatchDir == null),
         'an NPU bundle runs with a dispatch directory, a CPU one without',
       );

  /// The `.litertlm` bundle core resolved for the active spec.
  final String modelPath;

  /// `<applicationCache>/flutter_edge_ai/emb` (path_provider's
  /// `getApplicationCacheDirectory()`), resolved in `describe()` — the worker
  /// isolate has no platform channels. The pass keeps one directory per bundle
  /// copy under it. A cache directory, so it is never backed up (iOS iCloud,
  /// Android Auto Backup's 25 MB quota); when the OS purges it, the next engine
  /// create rebuilds the weight cache.
  final String cacheBase;

  /// The media the caller declared. The pass refuses, from ModelInfo and
  /// before creating an engine, images on a bundle without a vision encoder
  /// and audio (1.16.0).
  final Set<EmbeddingModality> declaredMedia;

  /// Engine threads; null leaves the engine's default (about 4).
  final int? numThreads;

  /// Vision tokens per image (D25): 140 or 70, checked by `describe()`.
  final int visionTokensPerImage;

  /// The SoC a per-SoC bundle is compiled for, when `describe()` placed it on
  /// the NPU; null runs the bundle on CPU.
  final LiteRtLmNpuTarget? npuTarget;

  /// Where LiteRT loads the Qualcomm dispatch and QNN libraries from, prepared
  /// by `describe()` on the caller's isolate (Android extracts them through a
  /// platform channel, which this worker has none of). Set with [npuTarget].
  final String? npuDispatchDir;

  /// The backend the engine is created on.
  PreferredBackend get backend =>
      npuTarget == null ? PreferredBackend.cpu : PreferredBackend.npu;
}

/// The descriptor factory: a top-level tear-off, so it survives the trip into
/// core's worker isolate.
EmbeddingPass buildLiteRtLmEmbeddingPass(LiteRtLmEmbeddingConfig config) =>
    LiteRtLmEmbeddingPass(config);

/// Every C function the pass calls. A `libLiteRtLm` that exports fewer is
/// refused at load by [checkLiteRtLmEmbeddingSymbols], by name, instead of
/// failing on the first lookup somewhere inside the worker.
const liteRtLmEmbeddingSymbols = <String>[
  'litert_lm_clear_last_error',
  'litert_lm_get_last_error_code',
  'litert_lm_get_last_error_message',
  'litert_lm_loaded_file_create',
  'litert_lm_loaded_file_delete',
  'litert_lm_loaded_file_model_type',
  'litert_lm_loaded_file_embedding_dimension',
  'litert_lm_loaded_file_embedding_signature_selection',
  'litert_lm_loaded_file_vision_signature_selection',
  'litert_lm_embedding_engine_settings_create',
  'litert_lm_embedding_engine_settings_delete',
  'litert_lm_embedding_engine_settings_set_cache_dir',
  'litert_lm_embedding_engine_settings_set_max_input_length',
  'litert_lm_embedding_engine_settings_set_num_threads',
  'litert_lm_embedding_engine_settings_set_vision_tokens_per_image',
  'litert_lm_embedding_engine_settings_set_litert_dispatch_lib_dir',
  'litert_lm_embedding_engine_settings_set_vision_litert_dispatch_lib_dir',
  'litert_lm_embedding_engine_settings_set_audio_litert_dispatch_lib_dir',
  'litert_lm_embedding_engine_create',
  'litert_lm_embedding_engine_delete',
  'litert_lm_embedding_engine_compute_embedding',
  'litert_lm_embedding_options_create',
  'litert_lm_embedding_options_delete',
  'litert_lm_embedding_options_set_normalize',
  'litert_lm_embedding_options_set_insert_special_tokens',
  'litert_lm_embedding_options_set_input_overflow_strategy',
  'litert_lm_input_data_create',
  'litert_lm_input_data_delete',
  'litert_lm_embedding_response_get_size',
  'litert_lm_embedding_response_get_values',
  'litert_lm_embedding_response_delete',
];

/// Throws [EmbeddingLoadException] naming every symbol of
/// [liteRtLmEmbeddingSymbols] that [provides] reports missing from [library].
void checkLiteRtLmEmbeddingSymbols(
  bool Function(String symbol) provides, {
  required String library,
}) {
  final missing = [
    for (final symbol in liteRtLmEmbeddingSymbols)
      if (!provides(symbol)) symbol,
  ];
  if (missing.isEmpty) return;
  throw EmbeddingLoadException(
    '$library does not export the LiteRT-LM EmbeddingEngine C API '
    '(missing: ${missing.join(', ')}). EmbeddingGemma 2 needs the native '
    'libraries of native-v0.18.0-d or newer; a stale local prebuilt under '
    'native/litert_lm/prebuilt/ is the usual cause.',
    engine: liteRtLmEngineTag,
  );
}

/// Binds [library] after checking it exports everything the pass calls.
LiteRtLmEmbeddingBindings bindLiteRtLmEmbeddings(
  DynamicLibrary library, {
  String label = 'libLiteRtLm',
}) {
  checkLiteRtLmEmbeddingSymbols(library.providesSymbol, library: label);
  return LiteRtLmEmbeddingBindings(library);
}

LiteRtLmEmbeddingBindings _openBindings() =>
    bindLiteRtLmEmbeddings(openLiteRtLmLibrary());

/// The `.litertlm` embedding pass: text, and images on a bundle with a vision
/// encoder; CPU, or the NPU for a per-SoC bundle.
///
/// Built and run inside one isolate: the native handles it holds cannot cross
/// an isolate boundary — only its [LiteRtLmEmbeddingConfig] does.
final class LiteRtLmEmbeddingPass extends EmbeddingPass {
  LiteRtLmEmbeddingPass(
    this._config, {
    @visibleForTesting LiteRtLmEmbeddingBindings Function()? openBindings,
    @visibleForTesting Allocator allocator = calloc,
    @visibleForTesting DateTime Function()? clock,
  }) : _open = openBindings ?? _openBindings,
       _alloc = allocator,
       _now = clock ?? DateTime.now;

  final LiteRtLmEmbeddingConfig _config;
  final LiteRtLmEmbeddingBindings Function() _open;

  /// Every pointer the pass allocates, and frees.
  final Allocator _alloc;

  /// What "now" is when judging whether a sibling cache is still in use.
  final DateTime Function() _now;

  LiteRtLmEmbeddingBindings? _bindings;
  Pointer<LiteRtLmEmbeddingEngine> _engine = nullptr;
  Pointer<LiteRtLmEmbeddingOptions> _options = nullptr;
  int _dimension = 0;
  int _maxInputTokens = 0;
  bool _images = false;
  String? _cacheDir;
  bool _loaded = false;
  bool _closed = false;

  String get _file => _fileName(_config.modelPath);

  /// The engine-owned cache directory [load] created; null before.
  @visibleForTesting
  String? get cacheDir => _cacheDir;

  @override
  Future<EmbeddingPassInfo> load() async {
    if (_loaded || _closed) {
      throw StateError('LiteRtLmEmbeddingPass.load() runs once, before close');
    }
    final LiteRtLmEmbeddingBindings b;
    try {
      b = _open();
    } on EmbeddingLoadException {
      rethrow;
    } catch (e) {
      throw EmbeddingLoadException(
        'could not open the LiteRT-LM native library: $e',
        engine: liteRtLmEngineTag,
        cause: e,
      );
    }
    _bindings = b;

    final info = _readModelInfo(b);
    final textSignatures = _checkModelInfo(info);
    final (:setting, :maxInputTokens) = liteRtLmInputLimit(textSignatures);
    final images = _config.declaredMedia.contains(EmbeddingModality.image);
    final cacheDir = _createCacheDir();

    try {
      _engine = _createEngine(
        b,
        maxInputLength: setting,
        cacheDir: cacheDir,
        // Only when images will be embedded: the budget means nothing to a
        // text request, and the bundle's own default is 140 anyway.
        visionTokensPerImage: images ? _config.visionTokensPerImage : null,
      );
      _options = _createOptions(b);
      if (_onNpu) _probeNpu(b, info.dimension);
    } catch (_) {
      _free();
      rethrow;
    }
    // Measured on CPU only: what an NPU engine caches (the CPU embedder's
    // weights, the compiled graph's context) is not established, so an empty
    // directory there is not reported as a fault.
    if (!_onNpu) _warnIfNoWeightCache(cacheDir);
    // Only now: a load that failed must not have cost another copy its cache.
    _deleteStaleSiblings(cacheDir);

    _dimension = info.dimension;
    _maxInputTokens = maxInputTokens;
    _images = images;
    _cacheDir = cacheDir;
    _loaded = true;
    // D23: what the BUNDLE can do. Core keeps the model at what was declared
    // (text, plus images only when the caller asked for them).
    final vision = _hasVision(info);
    edgeAiLog(
      '[LiteRtLmEmbeddingPass] loaded $_file: dim=${info.dimension}, '
      'maxInputTokens=$maxInputTokens, '
      'backend=${_onNpu ? 'npu (${_config.npuTarget}, dispatch '
                '${_config.npuDispatchDir})' : 'cpu'}, '
      'vision=${vision ? info.visionSignatures : 'none'}'
      '${images ? ', ${_config.visionTokensPerImage} tokens/image' : ''}, '
      'cache=$cacheDir',
    );
    return EmbeddingPassInfo(
      dimension: info.dimension,
      maxInputTokens: maxInputTokens,
      modalities: {EmbeddingModality.text, if (vision) EmbeddingModality.image},
      imageFormats: vision ? liteRtLmImageFormats : const {},
    );
  }

  @override
  Future<List<double>> embed(EmbeddingInput input, TaskType taskType) async {
    final b = _bindings;
    if (!_loaded || _closed || b == null) {
      throw const EmbeddingEngineException(
        'LiteRtLmEmbeddingPass.embed() needs a loaded, open pass',
        engine: liteRtLmEngineTag,
      );
    }
    final parts = embeddingParts(input, taskType);
    final kind = parts.length == 1 ? parts.single.kind : null;
    if (kind == null ||
        kind == EmbeddingPartKind.audio ||
        (kind == EmbeddingPartKind.image && !_images)) {
      // Core refuses all of these before sending (one modality per vector,
      // and only the modalities the model was loaded with); reaching here is
      // a host bug.
      throw EmbeddingEngineException(
        'flutter_edge_ai_litertlm embeds one text or one declared image per '
        'vector, got $input',
        engine: liteRtLmEngineTag,
      );
    }
    final image = kind == EmbeddingPartKind.image;

    // Everything below is synchronous: the native error state is
    // thread-local, and an `await` could resume this isolate on another
    // thread before it is read.
    final bytes = parts.single.bytes;
    final buffer = _alloc<Uint8>(math.max(bytes.length, 1));
    final Pointer<LiteRtLmInputData> part;
    try {
      buffer.asTypedList(bytes.length).setAll(0, bytes);
      // Copies the bytes (std::string). Returns NULL only for an unknown
      // type, and then sets no error — so none is read.
      part = b.litert_lm_input_data_create(
        image
            ? LiteRtLmInputDataType.kLiteRtLmInputDataTypeImage
            : LiteRtLmInputDataType.kLiteRtLmInputDataTypeText,
        buffer.cast(),
        bytes.length,
      );
    } finally {
      _alloc.free(buffer);
    }
    if (part == nullptr) {
      throw EmbeddingEngineException(
        'litert_lm_input_data_create returned NULL for '
        '${image ? 'an image' : 'a text'} part',
        engine: liteRtLmEngineTag,
      );
    }

    final partArray = _alloc<Pointer<LiteRtLmInputData>>();
    try {
      partArray[0] = part;
      // The error state is per thread and survives successful calls: clear
      // it, so a NULL with nothing reported is never read as an old status.
      final (:response, :code, :message) = _compute(b, partArray);
      if (response == nullptr) {
        if (code == _statusInvalidArgument && !image) {
          // §4.8: the only InvalidArgument a single, well-formed text part
          // built here can reach at v0.18.0 is the signature-length check.
          throw EmbeddingInputTooLongException(
            'The input is longer than the $_maxInputTokens tokens '
            '$_file accepts (task prefix, BOS and EOS included). Nothing was '
            'truncated; split the text and embed the pieces.',
            maxInputTokens: _maxInputTokens,
          );
        }
        throw EmbeddingEngineException(
          code == _statusInvalidArgument
              // An image cannot overflow at 140 tokens; stb refusing the
              // bytes is what InvalidArgument means for one (measured: a
              // truncated PNG, "outofdata").
              ? 'LiteRT-LM could not decode the image for $_file (a corrupt '
                    'or truncated file?): $message'
              : 'compute_embedding failed for $_file: $message',
          engine: liteRtLmEngineTag,
          nativeCode: code == _statusOk ? null : code,
        );
      }
      try {
        final size = b.litert_lm_embedding_response_get_size(response);
        final values = b.litert_lm_embedding_response_get_values(response);
        if (size != _dimension || values == nullptr) {
          throw EmbeddingEngineException(
            '$_file returned a vector of $size values for dimension '
            '$_dimension',
            engine: liteRtLmEngineTag,
          );
        }
        // The values belong to the response: copy before deleting it.
        return List<double>.generate(size, (i) => values[i]);
      } finally {
        b.litert_lm_embedding_response_delete(response);
      }
    } finally {
      b.litert_lm_input_data_delete(part);
      _alloc.free(partArray);
    }
  }

  @override
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    _free();
  }

  /// Deletes the options and the engine, whichever exist. Idempotent.
  void _free() {
    final b = _bindings;
    if (b == null) return;
    final options = _options;
    final engine = _engine;
    _options = nullptr;
    _engine = nullptr;
    if (options != nullptr) b.litert_lm_embedding_options_delete(options);
    if (engine != nullptr) {
      _npuCall(() => b.litert_lm_embedding_engine_delete(engine));
    }
  }

  bool get _onNpu => _config.npuTarget != null;

  /// Runs [body] as an NPU call must run on Linux: with SIGPROF blocked on
  /// this thread. A debug or profile VM samples every isolate thread with it,
  /// and Qualcomm's QNN backend does not survive that on Linux (measured on a
  /// QCS8275: engine create fails under the profiler, and works with the
  /// signal blocked). The LLM client blocks it around its engine create; this
  /// pass runs the engine synchronously on its worker's thread, so create,
  /// compute and delete all get it. Android's FastRPC survives the signal.
  T _npuCall<T>(T Function() body) =>
      _onNpu && Platform.isLinux ? withSigprofBlocked(body) : body();

  /// One `compute_embedding` of [parts], with the error copied in the same
  /// synchronous segment (and the same SIGPROF mask) as the call.
  ({Pointer<LiteRtLmEmbeddingResponse> response, int code, String message})
  _compute(
    LiteRtLmEmbeddingBindings b,
    Pointer<Pointer<LiteRtLmInputData>> parts,
  ) => _npuCall(() {
    // The error state is per thread and survives successful calls: clear it,
    // so a NULL with nothing reported is never read as an old status.
    b.litert_lm_clear_last_error();
    final response = b.litert_lm_embedding_engine_compute_embedding(
      _engine,
      parts,
      1,
      _options,
    );
    if (response != nullptr) return (response: response, code: 0, message: '');
    final (:code, :message) = _lastError(b);
    return (response: response, code: code, message: message);
  });

  /// The text the NPU load check embeds.
  static const _probeText = 'task: search result | query: probe';

  /// Embeds [_probeText] on a just-created NPU engine and throws
  /// [EmbeddingLoadException] unless it gets [dimension] finite values that
  /// are not all zero. `engine_create` does not run the graph — it reports a
  /// failed warm-up only in the native log — so without this a bundle the NPU
  /// cannot run (another SoC's, or a stack without this Hexagon version's
  /// Skel) would load and fail on the first request instead. Text only: the
  /// vision tower of a per-SoC bundle is compiled for the same SoC and loaded
  /// by the same dispatch, and probing it would cost an image embedding
  /// (about 0.3 s on an SM8650) at every load.
  void _probeNpu(LiteRtLmEmbeddingBindings b, int dimension) {
    final bytes = utf8.encode(_probeText);
    final buffer = _alloc<Uint8>(bytes.length);
    final Pointer<LiteRtLmInputData> part;
    try {
      buffer.asTypedList(bytes.length).setAll(0, bytes);
      part = b.litert_lm_input_data_create(
        LiteRtLmInputDataType.kLiteRtLmInputDataTypeText,
        buffer.cast(),
        bytes.length,
      );
    } finally {
      _alloc.free(buffer);
    }
    if (part == nullptr) {
      throw const EmbeddingLoadException(
        'litert_lm_input_data_create returned NULL for the NPU probe',
        engine: liteRtLmEngineTag,
      );
    }
    final partArray = _alloc<Pointer<LiteRtLmInputData>>();
    try {
      partArray[0] = part;
      final (:response, :code, :message) = _compute(b, partArray);
      if (response == nullptr) {
        throw EmbeddingLoadException(
          'LiteRT-LM could not run $_file on the NPU of this device: '
          '$message. The bundle is compiled for ${_config.npuTarget}; on '
          'another SoC, or with a QNN runtime that has no Skel for this '
          "device's Hexagon version, it fails this way.",
          engine: liteRtLmEngineTag,
          nativeCode: code == _statusOk ? null : code,
        );
      }
      try {
        final size = b.litert_lm_embedding_response_get_size(response);
        final values = b.litert_lm_embedding_response_get_values(response);
        final fault = size != dimension || values == nullptr
            ? '$size values for dimension $dimension'
            : _vectorFault(values, size);
        if (fault != null) {
          throw EmbeddingLoadException(
            '$_file on the ${_config.npuTarget} NPU returned $fault for a '
            'test input; it is not used.',
            engine: liteRtLmEngineTag,
          );
        }
      } finally {
        b.litert_lm_embedding_response_delete(response);
      }
    } finally {
      b.litert_lm_input_data_delete(part);
      _alloc.free(partArray);
    }
  }

  /// What is wrong with a vector of [size] [values], or null: a NaN or an
  /// infinity, or all zeros.
  static String? _vectorFault(Pointer<Float> values, int size) {
    var nonZero = false;
    for (var i = 0; i < size; i++) {
      final v = values[i];
      if (!v.isFinite) return 'a non-finite value ($v at $i)';
      if (v != 0) nonZero = true;
    }
    return nonZero ? null : 'an all-zero vector';
  }

  // ── ModelInfo ──────────────────────────────────────────────────────────

  _ModelInfo _readModelInfo(LiteRtLmEmbeddingBindings b) {
    final path = _config.modelPath.toNativeUtf8(allocator: _alloc);
    final Pointer<LiteRtLmLoadedFile> file;
    try {
      b.litert_lm_clear_last_error();
      file = b.litert_lm_loaded_file_create(path.cast());
    } finally {
      _alloc.free(path);
    }
    if (file == nullptr) {
      final (:code, :message) = _lastError(b);
      throw EmbeddingLoadException(
        'LiteRT-LM could not read $_file: $message',
        engine: liteRtLmEngineTag,
        nativeCode: code == _statusOk ? null : code,
      );
    }
    try {
      final LiteRtLmModelType? type;
      try {
        type = b.litert_lm_loaded_file_model_type(file);
      } on ArgumentError {
        // A model type newer than these bindings: not one this pass runs.
        return _unknownModelInfo;
      }
      return (
        type: type,
        dimension: b.litert_lm_loaded_file_embedding_dimension(file),
        textSignatures: _signatures(
          (lengths, max) =>
              b.litert_lm_loaded_file_embedding_signature_selection(
                file,
                lengths,
                max,
              ),
        ),
        visionSignatures: _signatures(
          (lengths, max) => b.litert_lm_loaded_file_vision_signature_selection(
            file,
            lengths,
            max,
          ),
        ),
      );
    } finally {
      b.litert_lm_loaded_file_delete(file);
    }
  }

  /// A ModelInfo length list: the count first (a NULL array), then the
  /// values. Null when the file defines none (-1).
  List<int>? _signatures(int Function(Pointer<Int32>, int) read) {
    final count = read(nullptr, 0);
    if (count < 0) return null;
    if (count == 0) return const [];
    final lengths = _alloc<Int32>(count);
    try {
      final written = math.min(read(lengths, count), count);
      return [for (var i = 0; i < written; i++) lengths[i]];
    } finally {
      _alloc.free(lengths);
    }
  }

  /// Refuses what this pass cannot run, from ModelInfo alone; returns the
  /// text signatures.
  List<int> _checkModelInfo(_ModelInfo info) {
    switch (info.type) {
      case LiteRtLmModelType.kLiteRtLmModelTypeEmbedding:
        break;
      case LiteRtLmModelType.kLiteRtLmModelTypeLlm:
        _refuse(
          '$_file is an LLM bundle, not an embedder. Install it with '
          '`installModel(fileType: ModelFileType.litertlm)` for inference, '
          'and an EmbeddingGemma 2 bundle with `installEmbedder()`.',
        );
      case LiteRtLmModelType.kLiteRtLmModelTypeUnknown || null:
        _refuse('$_file is not a LiteRT-LM embedding bundle.');
    }
    if (info.dimension <= 0) {
      _refuse('$_file declares no embedding dimension.');
    }
    final text = info.textSignatures;
    if (text == null || !text.any((length) => length > 0)) {
      _refuse('$_file declares no text signature to embed with.');
    }
    final media = _config.declaredMedia;
    if (media.contains(EmbeddingModality.audio)) {
      _refuse(liteRtLmNoAudioRefusal);
    }
    if (media.contains(EmbeddingModality.image)) {
      // D23: from the vision signatures, never supports_input_modality.
      if (!_hasVision(info)) {
        _refuse(
          '$_file has no vision encoder; install '
          'embeddinggemma-2-text-vision-440m',
        );
      }
      final budgets = info.visionSignatures!;
      final budget = _config.visionTokensPerImage;
      if (!budgets.contains(budget)) {
        _refuse(
          '$_file has vision signatures for ${budgets.join(' and ')} tokens '
          'per image, not $budget: '
          'LiteRtEmbeddingBackend(visionTokensPerImage: …) must be one of '
          'them.',
        );
      }
    }
    return text;
  }

  /// Whether the bundle has a vision encoder: a non-empty vision signature
  /// list (`[70, 140]` on the 440m and 740m, -1 on the 270m).
  static bool _hasVision(_ModelInfo info) =>
      info.visionSignatures?.any((length) => length > 0) ?? false;

  Never _refuse(String message) =>
      throw EmbeddingLoadException(message, engine: liteRtLmEngineTag);

  // ── Cache directory (D22) ──────────────────────────────────────────────

  /// Creates `<cacheBase>/<liteRtLmCacheDirName>/` for this copy of the
  /// bundle and returns its path. Nothing else is touched until the engine
  /// exists ([_deleteStaleSiblings]).
  String _createCacheDir() {
    try {
      final stat = File(_config.modelPath).statSync();
      if (stat.type == FileSystemEntityType.notFound) {
        throw FileSystemException('model file not found', _config.modelPath);
      }
      final name = liteRtLmCacheDirName(
        _file,
        size: stat.size,
        modified: stat.modified,
      );
      final dir = Directory(
        '${_config.cacheBase}${Platform.pathSeparator}$name',
      )..createSync(recursive: true);
      return dir.path;
    } on FileSystemException catch (e) {
      throw EmbeddingLoadException(
        'could not create the embedding cache directory for $_file under '
        '${_config.cacheBase}: ${e.osError?.message ?? e.message}',
        engine: liteRtLmEngineTag,
        cause: e,
      );
    }
  }

  /// Deletes the caches earlier copies of this file name left (a re-download
  /// or a new revision under the same name changes size or mtime). Runs only
  /// after this pass's engine exists, and spares any sibling touched within
  /// [_siblingGrace] — directory or newest entry — because another process
  /// (an app extension sharing the container, the app mid-replacement) may be
  /// creating its engine from it right now; it is collected by a later load.
  /// Costs disk, never correctness: a sibling that will not go is logged and
  /// the rest are still tried.
  void _deleteStaleSiblings(String cacheDir) {
    final keep = _fileName(cacheDir);
    final ours = RegExp(
      '^${liteRtLmCacheDirPrefix(_file)}-[0-9a-z]+-[0-9a-z]+\$',
    );
    final List<FileSystemEntity> entries;
    try {
      // Sorted, so the order is the same everywhere (listing order is not).
      entries = Directory(_config.cacheBase).listSync()
        ..sort((a, b) => a.path.compareTo(b.path));
    } on FileSystemException catch (e) {
      edgeAiLog(
        '[LiteRtLmEmbeddingPass] could not list ${_config.cacheBase}: $e',
      );
      return;
    }
    final now = _now();
    for (final entry in entries) {
      final name = _fileName(entry.path);
      if (entry is! Directory || name == keep || !ours.hasMatch(name)) {
        continue;
      }
      try {
        if (now.difference(_lastTouched(entry)) < _siblingGrace) continue;
        entry.deleteSync(recursive: true);
      } on FileSystemException catch (e) {
        edgeAiLog(
          '[LiteRtLmEmbeddingPass] could not delete the stale cache '
          '${entry.path}: $e',
        );
      }
    }
  }

  /// The latest modification of [dir] itself or of anything directly in it.
  static DateTime _lastTouched(Directory dir) {
    var latest = dir.statSync().modified;
    for (final entry in dir.listSync()) {
      final modified = entry.statSync().modified;
      if (modified.isAfter(latest)) latest = modified;
    }
    return latest;
  }

  /// How recently a sibling cache may have been touched and still be spared.
  static const _siblingGrace = Duration(minutes: 10);

  /// The engine writes its weight cache at create. When it could not (a
  /// path too long for the platform, a full disk) it runs anyway — slower to
  /// load and about 290 MB larger — and says so only in the native log. Said
  /// here in release builds too, because that is a cost the app pays silently.
  void _warnIfNoWeightCache(String cacheDir) {
    final bool empty;
    try {
      empty = Directory(cacheDir).listSync().whereType<File>().isEmpty;
    } on FileSystemException {
      return; // the directory existed a moment ago; nothing useful to add
    }
    if (!empty) return;
    // ignore: avoid_print
    print(
      '[flutter_edge_ai_litertlm] WARNING: LiteRT-LM wrote no weight cache to '
      '$cacheDir while loading $_file, so this embedder runs without one '
      '(slower to load, about 290 MB more memory). The native log says why.',
    );
  }

  // ── Settings, engine, options ──────────────────────────────────────────

  Pointer<LiteRtLmEmbeddingEngine> _createEngine(
    LiteRtLmEmbeddingBindings b, {
    required int maxInputLength,
    required String cacheDir,
    required int? visionTokensPerImage,
  }) {
    final path = _config.modelPath.toNativeUtf8(allocator: _alloc);
    final backend = (_onNpu ? 'npu' : 'cpu').toNativeUtf8(allocator: _alloc);
    final Pointer<LiteRtLmEmbeddingEngineSettings> settings;
    try {
      // Tower backends NULL (D26): present towers run on the main backend,
      // and naming one the bundle lacks fails create with code 5.
      b.litert_lm_clear_last_error();
      settings = b.litert_lm_embedding_engine_settings_create(
        path.cast(),
        backend.cast(),
        nullptr,
        nullptr,
      );
    } finally {
      _alloc.free(path);
      _alloc.free(backend);
    }
    if (settings == nullptr) {
      final (:code, :message) = _lastError(b);
      throw EmbeddingLoadException(
        'LiteRT-LM could not build embedding settings for $_file: $message',
        engine: liteRtLmEngineTag,
        nativeCode: code == _statusOk ? null : code,
      );
    }
    try {
      final dir = cacheDir.toNativeUtf8(allocator: _alloc);
      try {
        b.litert_lm_embedding_engine_settings_set_cache_dir(
          settings,
          dir.cast(),
        );
      } finally {
        _alloc.free(dir);
      }
      b.litert_lm_embedding_engine_settings_set_max_input_length(
        settings,
        maxInputLength,
      );
      final threads = _config.numThreads;
      if (threads != null) {
        b.litert_lm_embedding_engine_settings_set_num_threads(
          settings,
          threads,
        );
      }
      if (visionTokensPerImage != null) {
        // D25: 140 by default, 70 opt-in; checked against the bundle's vision
        // signatures from ModelInfo already.
        b.litert_lm_embedding_engine_settings_set_vision_tokens_per_image(
          settings,
          visionTokensPerImage,
        );
      }
      final dispatchDir = _config.npuDispatchDir;
      if (_onNpu && dispatchDir != null) {
        // Every executor: the towers run on the main backend, and each has
        // its own settings. Upstream skips a tower the bundle lacks.
        final dir = dispatchDir.toNativeUtf8(allocator: _alloc);
        try {
          b
            ..litert_lm_embedding_engine_settings_set_litert_dispatch_lib_dir(
              settings,
              dir.cast(),
            )
            ..litert_lm_embedding_engine_settings_set_vision_litert_dispatch_lib_dir(
              settings,
              dir.cast(),
            )
            ..litert_lm_embedding_engine_settings_set_audio_litert_dispatch_lib_dir(
              settings,
              dir.cast(),
            );
        } finally {
          _alloc.free(dir);
        }
      }
      final (:engine, :code, :message) = _npuCall(() {
        b.litert_lm_clear_last_error();
        final engine = b.litert_lm_embedding_engine_create(settings);
        if (engine != nullptr) return (engine: engine, code: 0, message: '');
        final (:code, :message) = _lastError(b);
        return (engine: engine, code: code, message: message);
      });
      if (engine == nullptr) {
        throw EmbeddingLoadException(
          _onNpu
              ? 'LiteRT-LM could not create the embedding engine for $_file '
                    'on the ${_config.npuTarget} NPU: $message'
              // Measured: a missing or read-only cache dir is code 3 at create.
              : code == _statusInvalidArgument
              ? 'LiteRT-LM refused the embedding cache directory $cacheDir '
                    '(missing or not writable): $message'
              : 'LiteRT-LM could not create the embedding engine for $_file: '
                    '$message',
          engine: liteRtLmEngineTag,
          nativeCode: code == _statusOk ? null : code,
        );
      }
      return engine;
    } finally {
      // The engine copied what it needs.
      b.litert_lm_embedding_engine_settings_delete(settings);
    }
  }

  Pointer<LiteRtLmEmbeddingOptions> _createOptions(
    LiteRtLmEmbeddingBindings b,
  ) {
    final options = b.litert_lm_embedding_options_create();
    if (options == nullptr) {
      // `new` in C++; a NULL here sets no error.
      throw const EmbeddingLoadException(
        'litert_lm_embedding_options_create returned NULL',
        engine: liteRtLmEngineTag,
      );
    }
    b
      ..litert_lm_embedding_options_set_normalize(options, true)
      // Pinned (D26): BOS 2 / EOS 1, as Sentence Transformers; off changes
      // the vectors (cos ≈ 0.90).
      ..litert_lm_embedding_options_set_insert_special_tokens(options, true)
      ..litert_lm_embedding_options_set_input_overflow_strategy(
        options,
        LiteRtLmInputOverflowStrategy.kLiteRtLmInputOverflowStrategyError,
      );
    // output_size is never set: MRL is an app recipe (D18).
    return options;
  }
}

/// The text input setting and the resulting limit for a bundle whose text
/// signatures are [textSignatures] (at least one positive).
///
/// The setting is [liteRtLmMaxInputLength] (D21), or the largest signature
/// when none reaches it; the engine then accepts inputs up to the smallest
/// signature at or above the setting — that is `maxInputTokens`.
({int setting, int maxInputTokens}) liteRtLmInputLimit(
  List<int> textSignatures,
) {
  final lengths = [
    for (final length in textSignatures)
      if (length > 0) length,
  ]..sort();
  if (lengths.isEmpty) {
    throw ArgumentError.value(textSignatures, 'textSignatures', 'no length');
  }
  final setting = math.min(liteRtLmMaxInputLength, lengths.last);
  return (
    setting: setting,
    maxInputTokens: lengths.firstWhere((length) => length >= setting),
  );
}

/// The cache directory name for one copy of a bundle:
/// `<liteRtLmCacheDirPrefix(file)>-<size>-<mtime ms>`, both numbers in base 36,
/// so a re-downloaded or replaced file never reads the weight cache of another
/// (the engine's own cache file names carry the same pair).
///
/// Short on purpose (R5 D22 spells the file name out): the engine names its
/// cache file `<model file name>_<mtime>_<size>.<graph>.xnnpack_cache` inside
/// this directory, and on Windows the whole path must stay under MAX_PATH or
/// the engine runs without a weight cache. With `<cache>/flutter_edge_ai/emb/`
/// and this 24-character name the example app's path is 232 characters for a
/// 20-character user name; spelling the file name out made it about 284.
String liteRtLmCacheDirName(
  String modelFileName, {
  required int size,
  required DateTime modified,
}) =>
    '${liteRtLmCacheDirPrefix(modelFileName)}-${size.toRadixString(36)}-'
    '${modified.millisecondsSinceEpoch.toRadixString(36)}';

/// The part of [liteRtLmCacheDirName] that names the file: the first 8 hex
/// digits of the SHA-256 of its name. Directories with this prefix and another
/// size or time are stale copies of the same file.
String liteRtLmCacheDirPrefix(String modelFileName) =>
    sha256.convert(utf8.encode(modelFileName)).toString().substring(0, 8);

typedef _ModelInfo = ({
  LiteRtLmModelType? type,
  int dimension,
  List<int>? textSignatures,
  List<int>? visionSignatures,
});

const _ModelInfo _unknownModelInfo = (
  type: null,
  dimension: -1,
  textSignatures: null,
  visionSignatures: null,
);

/// The calling thread's last LiteRT-LM error, copied. Call it right after the
/// failing call, before any other native call or `await`, and clear the state
/// right before that call: success never resets it, so without the clear an
/// earlier failure on this (pooled) thread would read as this one's. Code 0
/// after a NULL means the call reported nothing.
({int code, String message}) _lastError(LiteRtLmEmbeddingBindings b) {
  final code = b.litert_lm_get_last_error_code();
  final message = b.litert_lm_get_last_error_message();
  return (
    code: code,
    message: code == _statusOk || message == nullptr
        ? 'no error reported (status $code)'
        : message.cast<Utf8>().toDartString(),
  );
}

String _fileName(String path) => path.split(RegExp(r'[/\\]')).last;
