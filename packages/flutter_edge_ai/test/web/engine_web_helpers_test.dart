// engine_web.dart's two helpers for engine web arms: resolveWebModelUrl and
// WebModelSourceResolver.activeInferenceSpec.
//
// Both live in libraries that import dart:js_interop, which the VM cannot load,
// so this runs in a browser and `tool/test_all.sh` (VM only) does not pick it
// up. Browser runs belong on the GCE VM (never a local browser):
//   flutter test test/web/engine_web_helpers_test.dart --platform chrome
//
// NOT covered by any automated test yet: resolveWebModelUrl's success path for
// an `opfs://` path (`opfs.getStream` → `Response.blob()` →
// `URL.createObjectURL`) and its `opfs == null` refusal. Both need a real
// streaming-mode (OPFS) install. The logic is token-identical to the
// flutter_edge_ai_onnx resolver it replaced (`resolveOnnxWebPath`), and the
// web drive suite of the embedding-API PR adds an embedder run under
// WebStorageMode.streaming that exercises it end to end.
@TestOn('browser')
library;

import 'package:flutter_edge_ai/engine_web.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai/src/di/service_registry.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  // The web test entrypoint registers no plugins: bind the test binding and
  // mock SharedPreferences before anything touches them.
  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() {
    SharedPreferences.setMockInitialValues({});
    ServiceRegistry.reset();
  });

  tearDown(ServiceRegistry.reset);

  group('resolveWebModelUrl', () {
    test('returns blob:, https: and relative paths unchanged', () async {
      // Anything that is not `opfs://` returns before the registry is read.
      for (final path in [
        'blob:https://example.com/0a1b2c',
        'https://huggingface.co/org/repo/resolve/main/model.onnx',
        'assets/models/tokenizer.json',
      ]) {
        expect(await resolveWebModelUrl(path), path);
      }
    });

    test('refuses an opfs:// path when streaming storage is off', () async {
      await ServiceRegistry.initialize(webStorageMode: WebStorageMode.cacheApi);
      await expectLater(
        resolveWebModelUrl('opfs://model.onnx'),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            contains('streaming storage is not enabled'),
          ),
        ),
      );
    });
  });

  group('WebModelSourceResolver.activeInferenceSpec', () {
    test('throws StateError when no inference model is active', () async {
      await ServiceRegistry.initialize();
      await expectLater(
        WebModelSourceResolver.forActiveModel().activeInferenceSpec(),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            contains('No active inference model set'),
          ),
        ),
      );
    });

    test(
      'returns the restored active spec after awaiting the restore',
      () async {
        // The case onnx's TransformersWebResolver depends on: an ONNX web
        // inference identity is fileless (Transformers.js resolves the repo
        // itself), so it restores from prefs without an installed file.
        const repoUrl =
            'https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct';
        SharedPreferences.setMockInitialValues({
          'active_inference_model_type': 'qwen',
          'active_inference_file_type': 'onnx',
          'active_inference_filename': 'Qwen2.5-0.5B-Instruct',
          'active_inference_source': 'network|$repoUrl',
        });
        await ServiceRegistry.initialize();

        final spec = await WebModelSourceResolver.forActiveModel()
            .activeInferenceSpec();

        expect(spec.name, 'Qwen2.5-0.5B-Instruct');
        expect(spec.fileType, ModelFileType.onnx);
        expect(spec.modelType, ModelType.qwen);
        expect(spec.modelSource, NetworkSource(repoUrl));
      },
    );
  });
}
