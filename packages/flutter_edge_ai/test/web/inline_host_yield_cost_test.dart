// MEASUREMENT, not a regression gate: what the request pump's zero-timer yield
// (W2: `await Future<void>.delayed(Duration.zero)` before every request) costs
// on the web page. Browsers clamp nested timers to >= 4 ms after a few levels,
// so on the inline host every request may pay ~4 ms however fast the pass is.
//
// A pass that answers at once (`Future.value`), behind core's inline host and
// facade: `embedAll` of 200 short texts, printed as total and per-request ms,
// next to 200 bare zero-timer awaits on the same page as the clamp baseline.
// Asserts only that everything completed; the numbers decide whether the
// yield needs another shape on web.
//
// Not run by tool/test_all.sh or CI, which run on the VM. Run:
//   flutter test --platform chrome test/web/inline_host_yield_cost_test.dart
@TestOn('browser')
library;

import 'package:flutter_edge_ai/src/embedding/common_embedding_model.dart';
import 'package:flutter_edge_ai/src/embedding/embedding_host_web.dart';
import 'package:flutter_edge_ai/src/embedding/embedding_input.dart';
import 'package:flutter_edge_ai/src/embedding/embedding_pass.dart';
import 'package:flutter_edge_ai/src/embedding/task_type.dart';
import 'package:flutter_test/flutter_test.dart';

/// Answers every request at once: whatever the measurement shows is the
/// host's and the facade's, not the engine's.
final class _InstantPass extends EmbeddingPass {
  @override
  Future<EmbeddingPassInfo> load() async =>
      const EmbeddingPassInfo(dimension: 4);

  @override
  Future<List<double>> embed(EmbeddingInput input, TaskType taskType) =>
      Future.value(List<double>.filled(4, 0.0));

  @override
  Future<void> close() async {}
}

EmbeddingPass _buildInstantPass(String _) => _InstantPass();

String _ms(Duration d, int n) =>
    '${d.inMilliseconds} ms total, '
    '${(d.inMicroseconds / n / 1000).toStringAsFixed(2)} ms per request';

void main() {
  test('embedAll of 200 short texts through the inline host (prints the '
      'timing)', () async {
    const count = 200;

    // Baseline: 200 sequential zero-timer awaits, nothing else.
    final bare = Stopwatch()..start();
    for (var i = 0; i < count; i++) {
      await Future<void>.delayed(Duration.zero);
    }
    bare.stop();

    final host = await startInlineEmbeddingHost(
      const EmbeddingPassDescriptor<String>(
        engineTag: 'Instant',
        factory: _buildInstantPass,
        config: 'instant',
        activeBackend: null,
      ),
      declaredMedia: const {},
      label: 'yield-cost',
    );
    final model = CommonEmbeddingModel.withHost(host, engine: 'Instant');
    final inputs = [
      for (var i = 0; i < count; i++) EmbeddingInput.text('text $i'),
    ];
    // Warm-up, so the first-call costs of the page are not in the number.
    await model.embedAll(inputs.take(10).toList(), taskType: TaskType.none);

    final timed = Stopwatch()..start();
    final vectors = await model.embedAll(inputs, taskType: TaskType.none);
    timed.stop();

    // ignore: avoid_print
    print(
      '[inline-host yield cost] embedAll of $count short texts: '
      '${_ms(timed.elapsed, count)}',
    );
    // ignore: avoid_print
    print(
      '[inline-host yield cost] $count bare zero-timer awaits: '
      '${_ms(bare.elapsed, count)}',
    );

    expect(vectors, hasLength(count));
    await model.close();
  }, timeout: const Timeout(Duration(minutes: 2)));
}
