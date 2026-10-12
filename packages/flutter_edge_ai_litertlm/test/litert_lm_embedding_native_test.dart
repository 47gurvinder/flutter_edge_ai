// The EmbeddingGemma 2 pass on the REAL libLiteRtLm (native-v0.18.0-d) and
// the real bundles, in the VM — the X1 check of R5 §10.1 at the FFI level:
// text through these Dart bindings reproduces the golden `vector`, which came
// from the same macOS arm64 dylib through Python ctypes.
//
// Opt-in, skipped unless both are present:
// - the host libLiteRtLm (a macOS dev box has it in the hook's download
//   cache after any example build);
// - EG2_270M_LITERTLM=<path to embeddinggemma-2-text-270m.litertlm> (the
//   pinned revision, sha256 2d079ee2…86fb). EG2_440M_LITERTLM (revision
//   e301f74d, sha256 92dcbea1…2eb0) adds the vision-bundle cases: images
//   through the Dart bindings on the example's ten generated scenes
//   (example/assets/eg2_vision), text→image and near-duplicate retrieval,
//   the accepted formats, a truncated PNG, and M10 — whether a text query
//   over images ranks better raw or with the retrieval-query prefix.
//
//   EG2_270M_LITERTLM=… flutter test test/litert_lm_embedding_native_test.dart

import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_edge_ai/engine.dart';
import 'package:flutter_edge_ai/engine_io.dart';
import 'package:flutter_edge_ai_litertlm/src/embedding/litert_lm_embedding_pass.dart';
import 'package:flutter_edge_ai_litertlm/src/ffi/litert_lm_embedding_bindings.dart';
import 'package:flutter_test/flutter_test.dart';

const _goldenPath =
    '../flutter_edge_ai/example/integration_test/goldens/eg2_text_golden.json';
const _images = '../flutter_edge_ai/example/assets/eg2_vision';

/// The ten scenes and their captions (the X2 measurement's).
const _scenes = {
  'circle': 'a red circle on a white background',
  'square': 'a blue square on a white background',
  'triangle': 'a green triangle on a white background',
  'bar_chart': 'a bar chart of quarterly revenue',
  'pie_chart': 'a pie chart showing market share',
  'line_chart': 'a line chart of temperature by month',
  'grocery_receipt':
      'a grocery store receipt listing milk, bread, eggs and apples',
  'taxi_receipt': 'a taxi receipt with an airport fee',
  'stop_sign': 'a red stop sign',
  'house_scene': 'a small house on green grass under a yellow sun',
};

Uint8List _image(String file) => File('$_images/$file').readAsBytesSync();

/// For each query, the index of its best document and the margin of the
/// right one over the best wrong one.
({int top1, double meanMargin, List<String> misses}) _rank(
  List<List<double>> queries,
  List<List<double>> documents,
  List<String> names,
) {
  var top1 = 0;
  var margins = 0.0;
  final misses = <String>[];
  for (var i = 0; i < queries.length; i++) {
    final scores = [for (final d in documents) _cos(queries[i], d)];
    final best = scores.indexOf(scores.reduce(math.max));
    if (best == i) {
      top1++;
    } else {
      misses.add('${names[i]}->${names[best]}');
    }
    final wrong = [
      for (var j = 0; j < scores.length; j++)
        if (j != i) scores[j],
    ].reduce(math.max);
    margins += scores[i] - wrong;
  }
  return (top1: top1, meanMargin: margins / queries.length, misses: misses);
}

String? _dylib() => firstExistingPath(
  hostNativeLibraryCandidates(
    libFileName: hostNativeLibraryFileName('LiteRtLm'),
  ),
);

String? _env(String name) {
  final value = Platform.environment[name];
  return value != null && value.isNotEmpty && File(value).existsSync()
      ? value
      : null;
}

double _cos(List<double> a, List<double> b) {
  var dot = 0.0, na = 0.0, nb = 0.0;
  for (var i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (math.sqrt(na) * math.sqrt(nb));
}

void main() {
  final dylib = _dylib();
  final bundle270m = _env('EG2_270M_LITERTLM');
  final bundle440m = _env('EG2_440M_LITERTLM');
  final skip = dylib == null
      ? 'no host libLiteRtLm in the download cache'
      : bundle270m == null
      ? 'EG2_270M_LITERTLM is not set'
      : null;

  late Directory cacheBase;
  LiteRtLmEmbeddingBindings bindings() =>
      bindLiteRtLmEmbeddings(DynamicLibrary.open(dylib!), label: dylib);

  LiteRtLmEmbeddingPass pass(
    String modelPath, {
    Set<EmbeddingModality> media = const {},
    int visionTokensPerImage = 140,
  }) => LiteRtLmEmbeddingPass(
    LiteRtLmEmbeddingConfig(
      modelPath: modelPath,
      cacheBase: cacheBase.path,
      declaredMedia: media,
      visionTokensPerImage: visionTokensPerImage,
    ),
    openBindings: bindings,
  );

  setUp(() {
    cacheBase = Directory.systemTemp.createTempSync('eg2_native_cache_');
  });
  tearDown(() => cacheBase.deleteSync(recursive: true));

  test('the shipped library exports everything the pass calls', () {
    checkLiteRtLmEmbeddingSymbols(
      DynamicLibrary.open(dylib!).providesSymbol,
      library: dylib,
    );
  }, skip: dylib == null ? 'no host libLiteRtLm in the download cache' : null);

  test(
    '270m reproduces the golden vectors (X1), ranks every query\'s '
    'document first, and leaves its weight cache in its own directory',
    () async {
      final golden = jsonDecode(File(_goldenPath).readAsStringSync()) as Map;
      final items = (golden['items'] as List).cast<Map<String, dynamic>>();
      final p = pass(bundle270m!);
      final loadWatch = Stopwatch()..start();
      final info = await p.load();
      loadWatch.stop();
      try {
        expect(info.dimension, 768);
        expect(info.modalities, {EmbeddingModality.text});
        expect(info.maxInputTokens, 1024);

        final vectors = <String, List<double>>{};
        final timings = <int>[];
        var minGolden = 1.0, minSt = 1.0;
        for (final item in items) {
          final taskType = item['role'] == 'query'
              ? TaskType.retrievalQuery
              : TaskType.retrievalDocument;
          expect(taskType.prefix, item['prefix']);
          final watch = Stopwatch()..start();
          final vector = await p.embed(
            EmbeddingInput.text(item['text'] as String),
            taskType,
          );
          timings.add(watch.elapsedMicroseconds);
          final want = (item['vector'] as List)
              .cast<num>()
              .map((v) => v.toDouble())
              .toList();
          final st = (item['vector_st_fp32'] as List)
              .cast<num>()
              .map((v) => v.toDouble())
              .toList();
          final cosGolden = _cos(vector, want);
          final cosSt = _cos(vector, st);
          minGolden = math.min(minGolden, cosGolden);
          minSt = math.min(minSt, cosSt);
          expect(
            cosGolden,
            greaterThanOrEqualTo(0.9999),
            reason: '${item['id']}',
          );
          expect(cosSt, greaterThanOrEqualTo(0.97), reason: '${item['id']}');
          vectors[item['id'] as String] = vector;
        }
        for (var i = 0; i < 6; i++) {
          final query = vectors['q$i']!;
          final scores = [
            for (var j = 0; j < 6; j++) _cos(query, vectors['d$j']!),
          ];
          final best = scores.indexOf(scores.reduce(math.max));
          expect(best, i, reason: 'q$i ranks d$best first: $scores');
        }
        timings.sort();
        // ignore: avoid_print
        print(
          'EG2 270m on ${Platform.operatingSystem}: load '
          '${loadWatch.elapsedMilliseconds} ms; min cos vs golden '
          '$minGolden, vs ST fp32 $minSt; per embed median '
          '${timings[timings.length ~/ 2] / 1000} ms, max '
          '${timings.last / 1000} ms',
        );

        final dir = Directory(p.cacheDir!);
        expect(dir.parent.path, cacheBase.path);
        expect(
          dir.listSync().whereType<File>().any(
            (f) => f.path.endsWith('.xnnpack_cache'),
          ),
          isTrue,
          reason: 'the engine writes its weight cache where the pass told it',
        );
      } finally {
        await p.close();
      }
    },
    skip: skip,
  );

  test('an over-long text is EmbeddingInputTooLongException(1024, null), and '
      'the engine keeps working', () async {
    final p = pass(bundle270m!);
    await p.load();
    try {
      await expectLater(
        p.embed(
          EmbeddingInput.text('lorem ipsum dolor sit amet ' * 2300),
          TaskType.retrievalDocument,
        ),
        throwsA(
          isA<EmbeddingInputTooLongException>()
              .having((e) => e.maxInputTokens, 'maxInputTokens', 1024)
              .having((e) => e.actualTokens, 'actualTokens', isNull),
        ),
      );
      final vector = await p.embed(
        EmbeddingInput.text('still working'),
        TaskType.retrievalQuery,
      );
      expect(vector, hasLength(768));
    } finally {
      await p.close();
    }
  }, skip: skip);

  test('images declared on the 270m: no vision encoder, refused from '
      'ModelInfo', () async {
    final p = pass(bundle270m!, media: {EmbeddingModality.image});
    await expectLater(
      p.load(),
      throwsA(
        isA<EmbeddingLoadException>().having(
          (e) => e.message,
          'message',
          contains('has no vision encoder'),
        ),
      ),
    );
    await p.close();
    expect(cacheBase.listSync(), isEmpty, reason: 'no engine, no cache');
  }, skip: skip);

  test('an LLM bundle is not an embedder', () async {
    final p = pass('test/ffi/fixtures/gemma3_1b_prefix.litertlm');
    await expectLater(
      p.load(),
      throwsA(
        isA<EmbeddingLoadException>().having(
          (e) => e.message,
          'message',
          contains('is an LLM bundle, not an embedder'),
        ),
      ),
    );
    await p.close();
  }, skip: dylib == null ? 'no host libLiteRtLm in the download cache' : null);

  test(
    '440m: images at 140 tokens — text→image and near-duplicate top-1 on the '
    'ten scenes, GIF and BMP decode, a truncated PNG is code 3, M10; text '
    'still matches the 270m golden',
    () async {
      final golden = jsonDecode(File(_goldenPath).readAsStringSync()) as Map;
      final item = (golden['items'] as List).first as Map<String, dynamic>;
      final p = pass(bundle440m!, media: {EmbeddingModality.image});
      final loadWatch = Stopwatch()..start();
      final info = await p.load();
      loadWatch.stop();
      try {
        expect(info.modalities, {
          EmbeddingModality.text,
          EmbeddingModality.image,
        });
        expect(info.imageFormats, liteRtLmImageFormats);
        final text = await p.embed(
          EmbeddingInput.text(item['text'] as String),
          TaskType.retrievalQuery,
        );
        final want = (item['vector'] as List)
            .cast<num>()
            .map((v) => v.toDouble())
            .toList();
        expect(_cos(text, want), greaterThanOrEqualTo(0.9999));

        final names = _scenes.keys.toList();
        final images = <List<double>>[];
        final micros = <int>[];
        for (final name in names) {
          final watch = Stopwatch()..start();
          final vector = await p.embed(
            EmbeddingInput.image(_image('$name.png')),
            TaskType.retrievalDocument,
          );
          micros.add(watch.elapsedMicroseconds);
          expect(vector, hasLength(768));
          expect(vector.every((v) => v.isFinite), isTrue);
          images.add(vector);
        }
        final duplicates = [
          for (final name in names)
            await p.embed(
              EmbeddingInput.image(_image('${name}_dup.jpg')),
              TaskType.none,
            ),
        ];
        Future<List<List<double>>> captions(TaskType taskType) async => [
          for (final caption in _scenes.values)
            await p.embed(EmbeddingInput.text(caption), taskType),
        ];
        final raw = _rank(await captions(TaskType.none), images, names);
        final query = _rank(
          await captions(TaskType.retrievalQuery),
          images,
          names,
        );
        final dup = _rank(duplicates, images, names);

        // Formats: GIF and BMP decode to the same picture as the PNG.
        final small = await p.embed(
          EmbeddingInput.image(_image('small_circle.png')),
          TaskType.none,
        );
        final gif = await p.embed(
          EmbeddingInput.image(_image('small_circle.gif')),
          TaskType.none,
        );
        final bmp = await p.embed(
          EmbeddingInput.image(_image('small_circle.bmp')),
          TaskType.none,
        );
        expect(_cos(bmp, small), greaterThan(0.999), reason: 'BMP is lossless');
        expect(_cos(gif, small), greaterThan(0.98), reason: 'GIF is paletted');

        // A truncated PNG reaches the engine (the sniff reads only the
        // header) and fails to decode: code 3, never an overflow.
        final png = _image('circle.png');
        await expectLater(
          p.embed(
            EmbeddingInput.image(
              Uint8List.sublistView(png, 0, png.length ~/ 2),
            ),
            TaskType.none,
          ),
          throwsA(
            isA<EmbeddingEngineException>().having(
              (e) => e.nativeCode,
              'nativeCode',
              3,
            ),
          ),
        );
        expect(
          _cos(
            await p.embed(EmbeddingInput.image(png), TaskType.none),
            images.first,
          ),
          greaterThan(0.9999),
          reason: 'the engine keeps working after a failed decode',
        );

        micros.sort();
        // ignore: avoid_print
        print(
          'EG2 440m images on ${Platform.operatingSystem}: load '
          '${loadWatch.elapsedMilliseconds} ms; per image (140 tokens) median '
          '${micros[micros.length ~/ 2] / 1000} ms, max ${micros.last / 1000} '
          'ms; text→image raw top-1 ${raw.top1}/10 margin '
          '${raw.meanMargin.toStringAsFixed(4)} ${raw.misses}; with the query '
          'prefix top-1 ${query.top1}/10 margin '
          '${query.meanMargin.toStringAsFixed(4)} ${query.misses}; '
          'near-duplicate top-1 ${dup.top1}/10; cos(gif, png) '
          '${_cos(gif, small).toStringAsFixed(4)}, cos(bmp, png) '
          '${_cos(bmp, small).toStringAsFixed(4)}',
        );
        expect(raw.top1, 10, reason: '${raw.misses}');
        expect(dup.top1, 10, reason: '${dup.misses}');

        final dir = Directory(p.cacheDir!);
        final caches = {
          for (final f in dir.listSync().whereType<File>())
            f.path.split('.').reversed.elementAt(1): f.lengthSync(),
        };
        // ignore: avoid_print
        print('EG2 440m caches (bytes): $caches');
        expect(
          caches.keys,
          containsAll(['vision_encoder', 'text_encoder']),
          reason: 'the vision cache is written at create, into the pass dir',
        );
      } finally {
        await p.close();
      }
    },
    skip: skip ?? (bundle440m == null ? 'EG2_440M_LITERTLM is not set' : null),
  );

  test(
    '440m at 70 tokens per image: other vectors, about twice as fast',
    () async {
      final p = pass(
        bundle440m!,
        media: {EmbeddingModality.image},
        visionTokensPerImage: 70,
      );
      await p.load();
      final p140 = pass(bundle440m, media: {EmbeddingModality.image});
      try {
        final png = _image('circle.png');
        await p.embed(EmbeddingInput.image(png), TaskType.none); // warm up
        final watch = Stopwatch()..start();
        final at70 = await p.embed(EmbeddingInput.image(png), TaskType.none);
        watch.stop();
        await p.close();
        await p140.load();
        final at140 = await p140.embed(
          EmbeddingInput.image(png),
          TaskType.none,
        );
        final cos = _cos(at70, at140);
        // ignore: avoid_print
        print(
          'EG2 440m at 70 tokens: ${watch.elapsedMilliseconds} ms per image; '
          'cos(70, 140) ${cos.toStringAsFixed(4)}',
        );
        expect(cos, lessThan(0.9999), reason: 'the budget changes the vector');
        expect(cos, greaterThan(0.95));
      } finally {
        await p.close();
        await p140.close();
      }
    },
    skip: skip ?? (bundle440m == null ? 'EG2_440M_LITERTLM is not set' : null),
  );

  test(
    '440m with audio declared: refused, 1.15.0 embeds text and images',
    () async {
      final refused = pass(bundle440m!, media: {EmbeddingModality.audio});
      await expectLater(
        refused.load(),
        throwsA(
          isA<EmbeddingLoadException>().having(
            (e) => e.message,
            'message',
            liteRtLmNoAudioRefusal,
          ),
        ),
      );
      await refused.close();
    },
    skip: skip ?? (bundle440m == null ? 'EG2_440M_LITERTLM is not set' : null),
  );
}
