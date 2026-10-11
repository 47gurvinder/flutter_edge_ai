// TRANSITIONAL(C stage 3): remove with CommonEmbeddingModel.hostLegacyModel.
//
// A model an engine built through the pre-2.2 createModel seam reaches the app
// wrapped in CommonEmbeddingModel on the inline host: the 2.2 API over the old
// methods, core's checks, core's close — and a legacy model that closes itself
// still evicts (W6).

// The legacy model overrides the deprecated methods; that is the point.
// ignore_for_file: deprecated_member_use_from_same_package

import 'dart:async';
import 'dart:typed_data';

import 'package:flutter_edge_ai/src/domain/platform_types.dart';
import 'package:flutter_edge_ai/src/embedding/common_embedding_model.dart';
import 'package:flutter_edge_ai/src/embedding/embedding_exceptions.dart';
import 'package:flutter_edge_ai/src/embedding/embedding_input.dart';
import 'package:flutter_edge_ai/src/embedding/embedding_model.dart';
import 'package:flutter_edge_ai/src/embedding/task_type.dart';
import 'package:flutter_edge_ai/src/lifecycle/close_notifier.dart';
import 'package:flutter_test/flutter_test.dart';

/// The shape of litertlm's and onnx's pre-2.2 web models: the old methods,
/// no `embed`, no `dimension`.
class _LegacyModel extends EmbeddingModel with CloseNotifier {
  _LegacyModel({this.dimensionToReport = 3});

  final int dimensionToReport;
  final calls = <(String, TaskType)>[];
  int closeCalls = 0;
  bool _closed = false;

  @override
  PreferredBackend? get activeBackend => PreferredBackend.cpu;

  @override
  bool get isClosed => _closed;

  @override
  Future<List<double>> generateEmbedding(
    String text, {
    TaskType taskType = TaskType.retrievalQuery,
  }) async {
    calls.add((text, taskType));
    return List<double>.filled(dimensionToReport, 1);
  }

  @override
  Future<List<List<double>>> generateEmbeddings(
    List<String> texts, {
    TaskType taskType = TaskType.retrievalQuery,
  }) async => [for (final t in texts) await generateEmbedding(t)];

  @override
  Future<int> getDimension() async => dimensionToReport;

  @override
  Future<void> close() async {
    closeCalls++;
    if (_closed) return;
    _closed = true;
    fireCloseListeners();
  }
}

Future<CommonEmbeddingModel> _host(_LegacyModel legacy) =>
    CommonEmbeddingModel.hostLegacyModel(
      legacy,
      engineTag: 'Legacy',
      label: '/models/legacy.tflite',
    );

void main() {
  test('gets the 2.2 API over the old methods', () async {
    final legacy = _LegacyModel();
    final model = await _host(legacy);

    expect(model.dimension, 3);
    expect(model.maxInputTokens, isNull);
    expect(model.activeBackend, PreferredBackend.cpu);
    expect(
      await model.embed(
        EmbeddingInput.text('hi'),
        taskType: TaskType.retrievalDocument,
      ),
      [1.0, 1.0, 1.0],
    );
    expect(legacy.calls.single, ('hi', TaskType.retrievalDocument));
    await model.close();
  });

  test('media are refused before the legacy model sees them', () async {
    final legacy = _LegacyModel();
    final model = await _host(legacy);
    await expectLater(
      model.embed(
        EmbeddingInput.image(Uint8List.fromList([0xFF, 0xD8, 0xFF, 0])),
        taskType: TaskType.none,
      ),
      throwsA(isA<EmbeddingModalityNotLoadedException>()),
    );
    expect(legacy.calls, isEmpty);
    await model.close();
  });

  test(
    'task types newer than the legacy engine are refused, not guessed',
    () async {
      final legacy = _LegacyModel();
      final model = await _host(legacy);
      await expectLater(
        model.embed(EmbeddingInput.text('x'), taskType: TaskType.none),
        throwsA(
          isA<EmbeddingEngineException>()
              .having((e) => e.engine, 'engine', 'Legacy')
              .having((e) => e.message, 'message', contains('TaskType.none')),
        ),
      );
      expect(legacy.calls, isEmpty);
      await model.close();
    },
  );

  test('close closes the legacy model once', () async {
    final legacy = _LegacyModel();
    final model = await _host(legacy);
    var fired = 0;
    model.addCloseListener(() => fired++);

    await model.close();
    await model.close();
    expect(legacy.closeCalls, 1);
    expect(fired, 1);
    expect(model.isClosed, isTrue);
  });

  test('a legacy model that closes itself closes the wrapper (W6)', () async {
    final legacy = _LegacyModel();
    final model = await _host(legacy);
    var fired = 0;
    model.addCloseListener(() => fired++);

    await legacy.close(); // its worker died, say
    await pumpEventQueue();

    expect(model.isClosed, isTrue);
    expect(fired, 1);
    await expectLater(
      model.embed(EmbeddingInput.text('x'), taskType: TaskType.retrievalQuery),
      throwsA(isA<EmbedderClosedException>()),
    );
  });

  test('a load that fails closes the legacy model', () async {
    final legacy = _LegacyModel(dimensionToReport: 0);
    await expectLater(_host(legacy), throwsA(isA<EmbeddingLoadException>()));
    await pumpEventQueue();
    expect(legacy.closeCalls, 1);
  });
}
