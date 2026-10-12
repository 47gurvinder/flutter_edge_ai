// Records a voice memo in the Knowledge screen with the real microphone,
// models and storage: the memo is kept, transcribed by the active
// speech-to-text model and embedded as sound by EmbeddingGemma 2 (740M).
//
// The microphone must already be allowed — a permission dialog cannot be
// answered from a test. iOS Simulator:
//   xcrun simctl privacy <sim id> grant microphone dev.flutterberlin.flutterGemmaExample5
//   flutter test integration_test/knowledge_record_test.dart -d <sim id> \
//     --dart-define=KNOWLEDGE_EMBEDDER_FILE=... (see knowledge_models.dart)
// Android: grant android.permission.RECORD_AUDIO once the app is installed.
//
// No setUp/setUpAll (Rule 6b).

import 'package:flutter/material.dart';
import 'package:flutter_edge_ai_example/knowledge_screen.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'knowledge_models.dart';

void _log(String line) {
  // ignore: avoid_print
  print('[record] $line');
}

Future<void> _waitUntil(
  WidgetTester tester,
  bool Function() done,
  String what,
) async {
  final watch = Stopwatch()..start();
  while (!done()) {
    if (watch.elapsed > const Duration(minutes: 3)) {
      fail('timed out waiting for $what');
    }
    await tester.pump(const Duration(milliseconds: 100));
  }
}

void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  binding.framePolicy = LiveTestWidgetsFlutterBindingFramePolicy.fullyLive;

  testWidgets('a recorded memo is kept, transcribed and indexed', (
    tester,
  ) async {
    await ensureKnowledgeModels(log: _log, llm: false);
    await tester.pumpWidget(const MaterialApp(home: KnowledgeScreen()));
    await _waitUntil(
      tester,
      () => find.byKey(const Key('knowledge-index-card')).evaluate().isNotEmpty,
      'the indexes to open',
    );
    final record = find.byKey(const Key('knowledge-record'));
    await tester.scrollUntilVisible(
      record,
      400,
      scrollable: find
          .descendant(
            of: find.byKey(const Key('knowledge-library')),
            matching: find.byType(Scrollable),
          )
          .first,
    );
    await tester.tap(record);
    await _waitUntil(
      tester,
      () => find
          .byKey(const Key('knowledge-stop-recording'))
          .evaluate()
          .isNotEmpty,
      'the recording to start',
    );
    await tester.pump(const Duration(seconds: 3));
    await tester.tap(find.byKey(const Key('knowledge-stop-recording')));
    final memo = find.ancestor(
      of: find.textContaining('Voice memo 7'),
      matching: find.byType(Card),
    );
    await _waitUntil(
      tester,
      () =>
          memo.evaluate().isNotEmpty &&
          find
              .descendant(of: memo, matching: find.byIcon(Icons.check_circle))
              .evaluate()
              .isNotEmpty,
      'the memo to be indexed',
    );
    final texts = [
      for (final e
          in find.descendant(of: memo, matching: find.byType(Text)).evaluate())
        (e.widget as Text).data ?? '',
    ];
    _log('indexed: ${texts.join(' | ')}');
    expect(find.byKey(const Key('knowledge-error')), findsNothing);
  });
}
