// `createEmbeddingModel` on the two shells a VM test can construct: a new
// embedder is never built while the one it replaces is still closing, so two
// native models are never resident at once. Web is absent because
// `FlutterEdgeAiWeb` needs `dart:js_interop`; it uses the same `EmbedderCache`
// through the same `loadActiveEmbedder`.
//
// The backend here builds its own model (the pre-2.2 createModel seam), which
// core hosts; the gated close is that model's.
//
// Run: flutter test test/core/embedding/embedder_replacement_test.dart

import 'dart:async';

import 'package:flutter_edge_ai/src/lifecycle/close_notifier.dart';
import 'package:flutter_edge_ai/src/model_management/model_specs.dart';
import 'package:flutter_edge_ai/src/registry/embedding_backend_provider.dart';
import 'package:flutter_edge_ai/src/registry/embedding_registry.dart';
import 'package:flutter_edge_ai/src/registry/runtime_config.dart';
import 'package:flutter_edge_ai/src/interface.dart';
import 'package:flutter_test/flutter_test.dart';

import 'shell_fixture.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  final fixture = ShellFixture();
  late _CountingBackend backend;

  setUp(() async {
    await fixture.setUp();
    EmbeddingRegistry.instance.reset();
    backend = _CountingBackend();
    EmbeddingRegistry.instance.registerAll([backend]);
  });

  tearDown(() async {
    final gate = backend.closeGate;
    if (gate != null && !gate.isCompleted) gate.complete();
    await fixture.tearDown();
    EmbeddingRegistry.instance.reset();
  });

  for (final shell in shells) {
    test('${shell.name}: a replacement is not built until the embedder it '
        'replaces has finished closing', () async {
      final plugin = shell.create();
      final closeGate = Completer<void>();
      backend.closeGate = closeGate;

      await fixture.activate(plugin, model: 'a.tflite', tokenizer: 'a.json');
      await plugin.createEmbeddingModel();
      await fixture.activate(plugin, model: 'b.tflite', tokenizer: 'b.json');
      final second = plugin.createEmbeddingModel();
      await pumpEventQueue();

      expect(backend.models.first.teardowns, 1);
      expect(backend.builds, [
        fixture.modelPath('a.tflite'),
      ], reason: 'b must not be built while a is still closing');

      closeGate.complete();
      await second;
      expect(backend.builds, [
        fixture.modelPath('a.tflite'),
        fixture.modelPath('b.tflite'),
      ]);
    });

    test('${shell.name}: an embedder the app closed without awaiting is not '
        'rebuilt until its close has finished', () async {
      // `isClosed` is true from the moment close() is called. Taking that
      // alone as "gone" built the replacement while the app's own close was
      // still freeing the old native model.
      final plugin = shell.create();
      final closeGate = Completer<void>();
      backend.closeGate = closeGate;

      await fixture.activate(plugin, model: 'a.tflite', tokenizer: 'a.json');
      final first = await plugin.createEmbeddingModel();
      unawaited(first.close());

      final again = plugin.createEmbeddingModel();
      await pumpEventQueue();
      expect(backend.builds, [
        fixture.modelPath('a.tflite'),
      ], reason: 'the app\'s close of the first one is still running');

      closeGate.complete();
      final rebuilt = await again;
      expect(backend.builds, hasLength(2));
      expect(identical(rebuilt, first), isFalse);
      expect(
        backend.models.first.teardowns,
        1,
        reason: 'the shell joined the app\'s close rather than run another',
      );
    });
  }
}

/// Records every model path the shell asks it to build.
final class _CountingBackend extends EmbeddingBackendProvider {
  final List<String> builds = [];
  final List<_GatedCloseModel> models = [];

  /// When set, each built model's teardown waits on it.
  Completer<void>? closeGate;

  @override
  String get name => 'counting';

  @override
  bool canHandle(EmbeddingModelSpec spec) => true;

  @override
  Future<EmbeddingModel> createModel(
    EmbeddingModelSpec spec,
    RuntimeConfig config,
  ) async {
    builds.add(config.modelPath);
    final model = _GatedCloseModel(closeGate);
    models.add(model);
    return model;
  }
}

/// Closes the way a worker-backed model does: closed at once, torn down when
/// the gate opens, and every `close()` call returns the same teardown.
class _GatedCloseModel extends EmbeddingModel with CloseNotifier {
  _GatedCloseModel(this._closeGate);

  final Completer<void>? _closeGate;
  Future<void>? _teardown;
  bool _closed = false;
  int teardowns = 0;

  @override
  bool get isClosed => _closed;

  @override
  // ignore: deprecated_member_use_from_same_package
  Future<int> getDimension() async => 1;

  @override
  Future<void> close() => _teardown ??= _close();

  Future<void> _close() async {
    _closed = true;
    teardowns++;
    final gate = _closeGate;
    if (gate != null) await gate.future;
    fireCloseListeners();
  }
}
