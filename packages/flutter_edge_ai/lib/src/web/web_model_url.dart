// Resolves a web model path that core registered for an installed file into a
// URL `fetch()` (or an ONNX Runtime Web / Transformers.js loader) can open.
//
// Under `WebStorageMode.streaming` core registers `opfs://<filename>` paths
// (`WebDownloadService.registerUrl`). Neither `fetch()` nor
// `ort.InferenceSession.create()` understands that OPFS-private scheme, so the
// file is streamed out of OPFS into a `blob:` URL. Every other path
// (`WebStorageMode.cacheApi` / `.none`: already a `blob:` or `https:` URL) is
// returned unchanged.
//
// Same `ServiceRegistry.useStreamingStorage` + `WebDownloadService.opfsService`
// lookup `WebModelSourceResolver` does for inference models, but it produces a
// URL instead of a stream because its consumers want a URL. Moved from
// flutter_edge_ai_onnx's `resolveOnnxWebPath`, which reached into core's
// service registry to do it. Exported by `package:flutter_edge_ai/engine_web.dart`.
import 'dart:js_interop';

import 'package:flutter_edge_ai/src/di/service_registry.dart';
import 'package:flutter_edge_ai/src/infrastructure/web_download_service.dart';

@JS('Response')
extension type _JSResponse._(JSObject _) implements JSObject {
  external factory _JSResponse(JSAny body);
  external JSPromise<JSAny> blob();
}

@JS('URL.createObjectURL')
external JSString _createObjectUrlJs(JSAny blob);

/// Resolves [rawPath] — a model or tokenizer path core registered for an
/// installed file — to a URL that `fetch()` can load.
///
/// `opfs://<filename>` (streaming storage) becomes a `blob:` URL of that OPFS
/// file; any other path is returned unchanged. Throws [StateError] when an
/// `opfs://` path arrives while streaming storage is not enabled or OPFS is not
/// available.
///
/// Ownership: the `blob:` URL made for an `opfs://` path is a NEW object URL
/// that belongs to the caller. Revoke it (`URL.revokeObjectURL`) once its
/// consumer has read the file — on success and on failure alike — or it pins a
/// model-sized Blob until the page unloads. Any other path is returned as it
/// came and is not the caller's: never revoke it (core may own it).
Future<String> resolveWebModelUrl(String rawPath) async {
  if (!rawPath.startsWith('opfs://')) return rawPath;

  final registry = ServiceRegistry.instance;
  if (!registry.useStreamingStorage) {
    throw StateError(
      'Got an "opfs://" path ("$rawPath") but streaming storage is not '
      'enabled (webStorageMode != WebStorageMode.streaming).',
    );
  }
  final downloadService = registry.downloadService;
  if (downloadService is! WebDownloadService) {
    throw StateError(
      'Expected WebDownloadService to resolve "$rawPath" on web.',
    );
  }
  final opfs = downloadService.opfsService;
  if (opfs == null) {
    throw StateError(
      'OPFS service not available for "$rawPath" (streaming mode requires '
      'OPFS).',
    );
  }

  final filename = rawPath.substring('opfs://'.length);
  final stream = await opfs.getStream(filename);
  // `WebOPFSService.getStream` types as `Never` under the non-web analyzer
  // config (the `dart.library.js_interop` conditional-import stub) — the same
  // false positive `WebDownloadService` silences at its own `opfsService!`
  // call sites. Real web builds resolve the actual `Future<JSAny>`.
  // ignore: dead_code
  final blob = await _JSResponse(stream).blob().toDart;
  return _createObjectUrlJs(blob).toDart;
}
