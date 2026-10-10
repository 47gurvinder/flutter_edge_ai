// engine_web.dart's two helpers for engine web arms: resolveWebModelUrl and
// WebModelSourceResolver.activeInferenceSpec.
//
// Both live in libraries that import dart:js_interop, which the VM cannot load,
// so this runs in a browser and `tool/test_all.sh` (VM only) does not pick it
// up. Browser runs belong on the GCE VM (never a local browser):
//   flutter test test/web/engine_web_helpers_test.dart --platform chrome
//
// The OPFS → blob: branch of resolveWebModelUrl needs a real streamed install;
// it is exercised by the ONNX web embedding suites under WebStorageMode.streaming.
@TestOn('browser')
library;

import 'package:flutter_edge_ai/engine_web.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai/src/di/service_registry.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  setUp(ServiceRegistry.reset);
  tearDown(ServiceRegistry.reset);

  group('resolveWebModelUrl', () {
    test('returns blob:, https: and relative paths unchanged', () async {
      await ServiceRegistry.initialize();
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
      SharedPreferences.setMockInitialValues({});
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
        // A builtIn identity restores without an installed file, which keeps
        // this test free of downloads (see web_builtin_restore_test.dart).
        SharedPreferences.setMockInitialValues({
          'active_inference_model_type': 'general',
          'active_inference_file_type': 'builtIn',
          'active_inference_filename': 'gemini-nano',
          'active_inference_source': 'bundled|gemini-nano',
        });
        await ServiceRegistry.initialize();

        final spec = await WebModelSourceResolver.forActiveModel()
            .activeInferenceSpec();

        expect(spec.name, 'gemini-nano');
        expect(spec.fileType, ModelFileType.builtIn);
        expect(spec.modelSource, BundledSource('gemini-nano'));
      },
    );
  });
}
