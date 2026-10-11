// The shells' embedder path (R5 §4.2, §5.5, §10.1 "Shells"), driven through
// the real mobile and desktop shells over an installed ACTIVE embedder — since
// 2.2.0 there are no explicit paths. Covers what the backend is handed
// (RuntimeConfig: a null tokenizer path for a self-contained spec, the media
// flags, widened on a rebuild), the describe() path through core's own host,
// and the reuse / rebuild / concurrency rules each shell must keep by wrapping
// its entry point in EmbedderCache.serialize.

import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_edge_ai/src/embedding/embedding_exceptions.dart';
import 'package:flutter_edge_ai/src/embedding/embedding_input.dart';
import 'package:flutter_edge_ai/src/embedding/embedding_modality.dart';
import 'package:flutter_edge_ai/src/embedding/embedding_pass.dart';
import 'package:flutter_edge_ai/src/embedding/task_type.dart';
import 'package:flutter_edge_ai/src/lifecycle/close_notifier.dart';
import 'package:flutter_edge_ai/src/media/encoded_media_format.dart';
import 'package:flutter_edge_ai/src/mobile/flutter_edge_ai_mobile.dart';
import 'package:flutter_edge_ai/src/model_management/model_specs.dart';
import 'package:flutter_edge_ai/src/registry/embedding_backend_provider.dart';
import 'package:flutter_edge_ai/src/registry/embedding_registry.dart';
import 'package:flutter_edge_ai/src/registry/runtime_config.dart';
import 'package:flutter_edge_ai/src/domain/platform_types.dart';
import 'package:flutter_edge_ai/src/interface.dart';
import 'package:flutter_test/flutter_test.dart';

import 'shell_fixture.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  final fixture = ShellFixture();

  setUp(() async {
    await fixture.setUp();
    EmbeddingRegistry.instance.reset();
  });

  tearDown(() async {
    await fixture.tearDown();
    EmbeddingRegistry.instance.reset();
  });

  group('what the backend is handed', () {
    for (final shell in shells) {
      test(
        '${shell.name}: a self-contained spec gets a null tokenizer path',
        () async {
          final backend = _DescribingBackend();
          EmbeddingRegistry.instance.registerAll([backend]);
          final plugin = shell.create();

          await fixture.activate(plugin, model: 'eg2.litertlm');
          final model = await plugin.createEmbeddingModel();

          final config = backend.configs.single;
          expect(config.modelPath, fixture.modelPath('eg2.litertlm'));
          expect(config.tokenizerPath, isNull, reason: 'never the model path');
          expect(model.dimension, 3);
          await model.close();
        },
      );

      test('${shell.name}: a two-file spec gets its tokenizer path', () async {
        final backend = _DescribingBackend();
        EmbeddingRegistry.instance.registerAll([backend]);
        final plugin = shell.create();

        await fixture.activate(plugin, model: 'm.tflite', tokenizer: 't.json');
        await plugin.createEmbeddingModel();

        expect(backend.configs.single.tokenizerPath, endsWith('t.json'));
      });

      test(
        '${shell.name}: declared media reach the backend and the model',
        () async {
          final backend = _DescribingBackend(imageCapable: true);
          EmbeddingRegistry.instance.registerAll([backend]);
          final plugin = shell.create();

          await fixture.activate(plugin, model: 'eg2.litertlm');
          final model = await plugin.createEmbeddingModel(supportImage: true);

          expect(backend.configs.single.supportImage, isTrue);
          expect(backend.configs.single.supportAudio, isFalse);
          expect(model.modalities, {
            EmbeddingModality.text,
            EmbeddingModality.image,
          });
        },
      );

      test('${shell.name}: a text-only request reuses an image-capable '
          'embedder; a new medium rebuilds with the union', () async {
        final backend = _DescribingBackend(imageCapable: true);
        EmbeddingRegistry.instance.registerAll([backend]);
        final plugin = shell.create();

        await fixture.activate(plugin, model: 'eg2.litertlm');
        final image = await plugin.createEmbeddingModel(supportImage: true);
        final text = await plugin.createEmbeddingModel();
        expect(identical(image, text), isTrue);

        await expectLater(
          plugin.createEmbeddingModel(supportAudio: true),
          throwsA(
            isA<EmbeddingLoadException>().having(
              (e) => e.message,
              'message',
              contains('audio'),
            ),
          ),
          reason: 'the fake has no audio tower, and says so at load',
        );
        expect(backend.configs.last.supportImage, isTrue, reason: 'widened');
        expect(backend.configs.last.supportAudio, isTrue);
        expect(image.isClosed, isTrue, reason: 'one engine at a time');
      });
    }

    test('the union keeps media the request did not repeat, and the cache '
        'remembers it', () async {
      final backend = _DescribingBackend(
        imageCapable: true,
        audioCapable: true,
      );
      EmbeddingRegistry.instance.registerAll([backend]);
      final plugin = FlutterEdgeAiMobile();
      await fixture.activate(plugin, model: 'eg2.litertlm');

      await plugin.createEmbeddingModel(supportAudio: true);
      final both = await plugin.createEmbeddingModel(supportImage: true);
      expect(backend.configs, hasLength(2));
      expect(backend.configs.last.supportImage, isTrue);
      expect(
        backend.configs.last.supportAudio,
        isTrue,
        reason: 'audio was loaded before; a rebuild never narrows',
      );
      expect(both.modalities, {
        EmbeddingModality.text,
        EmbeddingModality.image,
        EmbeddingModality.audio,
      });

      // Each medium alone is now served by the widened embedder.
      expect(await plugin.createEmbeddingModel(supportAudio: true), same(both));
      expect(await plugin.createEmbeddingModel(supportImage: true), same(both));
      expect(backend.configs, hasLength(2), reason: 'no third build');
    });

    test('the describe() path hosts the engine pass: requests reach it, '
        'validated, with the task prefix decision left to the pass', () async {
      EmbeddingRegistry.instance.registerAll([_DescribingBackend()]);
      final plugin = FlutterEdgeAiMobile();

      await fixture.activate(plugin, model: 'eg2.litertlm');
      final model = await plugin.createEmbeddingModel(
        preferredBackend: PreferredBackend.gpu,
      );

      expect(model.activeBackend, PreferredBackend.cpu);
      expect(model.maxInputTokens, 64);
      expect(
        await model.embed(
          EmbeddingInput.text('abc'),
          taskType: TaskType.factChecking,
        ),
        [TaskType.factChecking.index.toDouble(), 3, 1],
      );
      await expectLater(
        model.embed(
          EmbeddingInput.image(Uint8List.fromList([0xFF, 0xD8, 0xFF, 0xE0])),
          taskType: TaskType.none,
        ),
        throwsA(isA<EmbeddingModalityNotLoadedException>()),
      );
      await model.close();
    });

    test('a pass still closing after its failed load holds the next build '
        '(never two engines resident)', () async {
      final gate = '${Directory.systemTemp.createTempSync('gate').path}/open';
      final backend = _DescribingBackend(failLoadGate: gate);
      EmbeddingRegistry.instance.registerAll([backend]);
      final plugin = FlutterEdgeAiMobile();
      await fixture.activate(plugin, model: 'eg2.litertlm');
      addTearDown(() {
        if (!File(gate).existsSync()) File(gate).writeAsStringSync('');
      });

      await expectLater(
        plugin.createEmbeddingModel(),
        throwsA(isA<EmbeddingLoadException>()),
      );
      while (!File('$gate.closing').existsSync()) {
        await Future<void>.delayed(const Duration(milliseconds: 5));
      }

      final retry = plugin.createEmbeddingModel();
      retry.ignore();
      await Future<void>.delayed(const Duration(milliseconds: 300));
      expect(
        backend.configs,
        hasLength(1),
        reason: 'the retry must wait for the failed pass to close',
      );

      File(gate).writeAsStringSync('');
      await expectLater(retry, throwsA(isA<EmbeddingLoadException>()));
      expect(backend.configs, hasLength(2));
    });

    test('a backend that refuses in describe() fails the load, typed, and '
        'leaves nothing cached', () async {
      final backend = _DescribingBackend(refuse: true);
      EmbeddingRegistry.instance.registerAll([backend]);
      final plugin = FlutterEdgeAiMobile();

      await fixture.activate(plugin, model: 'eg2.litertlm');
      await expectLater(
        plugin.createEmbeddingModel(supportImage: true),
        throwsA(isA<EmbeddingLoadException>()),
      );
      expect(plugin.initializedEmbeddingModel, isNull);
    });

    test('a backend that predates describe() cannot take media', () async {
      // TRANSITIONAL(C stage 3): remove with the createModel path.
      final backend = _LegacyBackend();
      EmbeddingRegistry.instance.registerAll([backend]);
      final plugin = FlutterEdgeAiMobile();

      await fixture.activate(plugin, model: 'm.tflite', tokenizer: 't.json');
      await expectLater(
        plugin.createEmbeddingModel(supportImage: true),
        throwsA(isA<EmbeddingLoadException>()),
      );
      expect(backend.built, isEmpty, reason: 'refused before createModel');

      // Text still works, wrapped in core's model.
      final model = await plugin.createEmbeddingModel();
      expect(model.dimension, 1);
      expect(backend.built, hasLength(1));
    });

    test('no backend that can run the spec is a StateError', () async {
      EmbeddingRegistry.instance.registerAll([
        _DescribingBackend(handles: false),
      ]);
      final plugin = FlutterEdgeAiMobile();
      await fixture.activate(plugin, model: 'eg2.litertlm');
      await expectLater(plugin.createEmbeddingModel(), throwsStateError);
    });
  });

  group('reuse, rebuild and concurrency (each shell serialises its entry)', () {
    late _LegacyBackend backend;

    setUp(() {
      backend = _LegacyBackend();
      EmbeddingRegistry.instance.registerAll([backend]);
    });

    test('the same active model is reused; a different one rebuilds', () async {
      final plugin = FlutterEdgeAiMobile();
      await fixture.activate(plugin, model: 'a.tflite', tokenizer: 'a.json');
      final first = await plugin.createEmbeddingModel();
      final again = await plugin.createEmbeddingModel();
      expect(identical(first, again), isTrue);
      expect(backend.built, hasLength(1));

      await fixture.activate(plugin, model: 'b.tflite', tokenizer: 'b.json');
      final second = await plugin.createEmbeddingModel();
      expect(identical(first, second), isFalse);
      expect(backend.built, hasLength(2));
      expect(backend.built.first.closed, isTrue);
    });

    test('a different tokenizer alone is a different embedder', () async {
      final plugin = FlutterEdgeAiMobile();
      await fixture.activate(
        plugin,
        model: 'a.tflite',
        tokenizer: 'sentencepiece.model',
      );
      await plugin.createEmbeddingModel();
      await fixture.activate(
        plugin,
        model: 'a.tflite',
        tokenizer: 'wordpiece.json',
      );
      await plugin.createEmbeddingModel();
      expect(backend.built, hasLength(2));
    });

    for (final shell in shells) {
      test(
        '${shell.name}: two concurrent first calls build ONE embedder',
        () async {
          final plugin = shell.create();
          await fixture.activate(
            plugin,
            model: 'a.tflite',
            tokenizer: 'a.json',
          );
          backend.gate = Completer<void>();

          final a = plugin.createEmbeddingModel();
          await Future<void>.delayed(Duration.zero);
          final b = plugin.createEmbeddingModel();
          // Long enough for b to resolve its paths and reach the backend if
          // nothing held it; serialised, it waits behind a's build.
          await Future<void>.delayed(const Duration(milliseconds: 200));
          expect(
            backend.paths,
            hasLength(1),
            reason: 'b must wait in the lane, not start a second build',
          );
          backend.gate!.complete();
          final models = await Future.wait([a, b]);

          expect(backend.built, hasLength(1));
          expect(identical(models[0], models[1]), isTrue);
        },
      );

      test('${shell.name}: a request for another model arriving mid-build is '
          'served its own model, and the superseded one is closed', () async {
        final plugin = shell.create();
        await fixture.activate(plugin, model: 'a.tflite', tokenizer: 'a.json');
        backend.gate = Completer<void>();

        final a = plugin.createEmbeddingModel();
        // Let a's request start and reach the backend before b is asked for.
        await pumpEventQueue();
        await fixture.activate(plugin, model: 'b.tflite', tokenizer: 'b.json');
        final b = plugin.createEmbeddingModel();

        backend.gate!.complete();
        final models = await Future.wait([a, b]);

        expect(backend.paths, [
          fixture.modelPath('a.tflite'),
          fixture.modelPath('b.tflite'),
        ]);
        expect(identical(models[0], models[1]), isFalse);
        expect(models[0].isClosed, isTrue, reason: 'b closed the superseded a');
        expect(plugin.initializedEmbeddingModel, same(models[1]));
      });

      test('${shell.name}: a failed build leaves no baseline behind', () async {
        final plugin = shell.create();
        await fixture.activate(plugin, model: 'a.tflite', tokenizer: 'a.json');
        backend.failNext = true;

        await expectLater(plugin.createEmbeddingModel(), throwsStateError);
        // The retry is a fresh decision, not a replay of the failure.
        await plugin.createEmbeddingModel();
        expect(backend.paths, hasLength(2));
      });
    }
  });
}

/// A backend that describes a pass core hosts in its worker isolate.
final class _DescribingBackend extends EmbeddingBackendProvider {
  _DescribingBackend({
    this.imageCapable = false,
    this.audioCapable = false,
    this.refuse = false,
    this.handles = true,
    this.failLoadGate,
  });

  /// When set, the pass fails its load and its close waits until this file
  /// exists (the pass runs in the worker isolate, so a file is the gate).
  final String? failLoadGate;

  final bool imageCapable;
  final bool audioCapable;
  final bool refuse;
  final bool handles;
  final configs = <RuntimeConfig>[];

  @override
  String get name => 'Describing';

  @override
  bool canHandle(EmbeddingModelSpec spec) => handles;

  @override
  Future<EmbeddingPassDescriptor<Object>> describe(
    EmbeddingModelSpec spec,
    RuntimeConfig config,
  ) async {
    configs.add(config);
    if (refuse) {
      throw const EmbeddingLoadException(
        'this engine refuses media',
        engine: 'Describing',
      );
    }
    return EmbeddingPassDescriptor<_PassConfig>(
      engineTag: 'Describing',
      factory: _buildPass,
      config: _PassConfig(
        imageCapable: imageCapable,
        audioCapable: audioCapable,
        failLoadGate: failLoadGate,
      ),
      activeBackend: PreferredBackend.cpu,
    );
  }
}

final class _PassConfig {
  const _PassConfig({
    required this.imageCapable,
    required this.audioCapable,
    this.failLoadGate,
  });
  final bool imageCapable;
  final bool audioCapable;
  final String? failLoadGate;
}

EmbeddingPass _buildPass(_PassConfig config) => _Pass(config);

final class _Pass extends EmbeddingPass {
  _Pass(this._config);

  final _PassConfig _config;

  @override
  Future<EmbeddingPassInfo> load() async {
    if (_config.failLoadGate != null) throw StateError('load refused');
    return _info();
  }

  EmbeddingPassInfo _info() => EmbeddingPassInfo(
    dimension: 3,
    maxInputTokens: 64,
    modalities: {
      EmbeddingModality.text,
      if (_config.imageCapable) EmbeddingModality.image,
      if (_config.audioCapable) EmbeddingModality.audio,
    },
    imageFormats: _config.imageCapable ? {EncodedMediaFormat.jpeg} : const {},
    audioFormats: _config.audioCapable ? {EncodedMediaFormat.wav} : const {},
  );

  @override
  Future<List<double>> embed(EmbeddingInput input, TaskType taskType) async => [
    taskType.index.toDouble(),
    (input.text ?? '').length.toDouble(),
    1,
  ];

  @override
  Future<void> close() async {
    final gate = _config.failLoadGate;
    if (gate == null) return;
    File('$gate.closing').writeAsStringSync('');
    while (!File(gate).existsSync()) {
      await Future<void>.delayed(const Duration(milliseconds: 5));
    }
  }
}

/// A backend that still builds its own model (the pre-2.2 seam).
// TRANSITIONAL(C stage 3): becomes a describing backend in the fake rewrite.
final class _LegacyBackend extends EmbeddingBackendProvider {
  final built = <_InertModel>[];
  final paths = <String>[];
  Completer<void>? gate;
  bool failNext = false;

  @override
  String get name => 'Legacy';

  @override
  bool canHandle(EmbeddingModelSpec spec) => true;

  @override
  Future<EmbeddingModel> createModel(
    EmbeddingModelSpec spec,
    RuntimeConfig config,
  ) async {
    paths.add(config.modelPath);
    if (gate != null) await gate!.future;
    if (failNext) {
      failNext = false;
      throw StateError('backend refused to build');
    }
    final model = _InertModel();
    built.add(model);
    return model;
  }
}

class _InertModel extends EmbeddingModel with CloseNotifier {
  bool closed = false;

  @override
  bool get isClosed => closed;

  @override
  // ignore: deprecated_member_use_from_same_package
  Future<int> getDimension() async => 1;

  @override
  Future<void> close() async {
    if (closed) return;
    closed = true;
    fireCloseListeners();
  }
}
