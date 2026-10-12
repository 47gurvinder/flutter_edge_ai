// The Knowledge screen's scripted demo: drives the REAL screen, with the real
// models, at a human pace, for a screen recording (about a minute):
//
//   home → Knowledge → index the library (notes, photos, voice memos; memos
//   transcribed first) → type "my flight is delayed" → a VOICE MEMO comes
//   first, and plays → tap "the airport taxi receipt" → a PHOTO comes first
//   → Ask "How much was the airport taxi, and will finance pay it back?" →
//   Gemma 4 reads the receipt photo and the trip note and answers, citing
//   both.
//
// It also checks what it shows: the memo and the photo rank first, the answer
// cites its sources, and it prints `[demo]` timing lines.
//
// Models (installed first, a no-op once installed; each becomes the active
// one of its kind):
// - EmbeddingGemma 2 Text + Vision + Audio (740M): --dart-define=
//   KNOWLEDGE_EMBEDDER_FILE=<a local copy> (checked against the catalog's
//   SHA-256), else downloaded (485 MB, ungated);
// - Gemma 4 E2B: KNOWLEDGE_LLM_FILE=<path to gemma-4-E2B-it.litertlm> (keep
//   that file name: it is how the catalog recognizes the model), else
//   downloaded (2.6 GB, ungated);
// - Moonshine Tiny: KNOWLEDGE_STT_MODEL_FILE + KNOWLEDGE_STT_TOKENIZER_FILE,
//   else downloaded (109 MB).
// Options: KNOWLEDGE_DEMO_PACE=1.5 (slower; 1.0 by default),
// KNOWLEDGE_DEMO_SHOTS=true (PNG screenshots into <documents>/knowledge_shots).
//
// Record it:
//   iOS Simulator (CPU there): xcrun simctl boot "iPhone 17 Pro"; then
//     xcrun simctl io booted recordVideo --codec h264 knowledge.mp4 &
//     flutter test integration_test/knowledge_demo_test.dart -d <sim id> \
//       --dart-define=KNOWLEDGE_LLM_FILE=... (a path the simulator can read)
//     and stop the recording (Ctrl-C / kill -INT) when the test passes.
//   Android emulator (arm64): adb push the model files to /data/local/tmp/,
//     adb shell screenrecord --time-limit 180 /sdcard/knowledge.mp4 &
//     flutter test integration_test/knowledge_demo_test.dart -d emulator-5554 \
//       --dart-define=KNOWLEDGE_LLM_FILE=/data/local/tmp/gemma-4-E2B-it.litertlm
//   macOS (phone-sized, 390 x 844, scaled into the window):
//     flutter test integration_test/knowledge_demo_test.dart -d macos ...
//
// No setUp/setUpAll (Rule 6b): the test body provisions first.

import 'dart:io';
import 'dart:ui' as ui;

import 'package:crypto/crypto.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_example/gemma_bootstrap.dart';
import 'package:flutter_edge_ai_example/knowledge/knowledge_app_dependencies.dart'
    show kKnowledgeLibraryKey;
import 'package:flutter_edge_ai_example/knowledge/knowledge_widgets.dart';
import 'package:flutter_edge_ai_example/main.dart';
import 'package:flutter_edge_ai_example/models/embedding_model.dart' as catalog;
import 'package:flutter_edge_ai_example/models/model.dart';
import 'package:flutter_edge_ai_example/models/stt_model.dart';
import 'package:flutter_edge_ai_example/services/embedding_catalog_install.dart';
import 'package:flutter_edge_ai_example/services/embedding_catalog_provenance.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _embedderFile = String.fromEnvironment('KNOWLEDGE_EMBEDDER_FILE');
const _llmFile = String.fromEnvironment('KNOWLEDGE_LLM_FILE');
const _sttModelFile = String.fromEnvironment('KNOWLEDGE_STT_MODEL_FILE');
const _sttTokenizerFile = String.fromEnvironment(
  'KNOWLEDGE_STT_TOKENIZER_FILE',
);
const _paceDefine = String.fromEnvironment('KNOWLEDGE_DEMO_PACE');
const _shots = bool.fromEnvironment('KNOWLEDGE_DEMO_SHOTS');

const _embedder = catalog.EmbeddingModel.embeddingGemma2TextVisionAudio740m;
const _llm = Model.gemma4_E2B_litertlm;
const _stt = SttModel.moonshineTiny;

final double _pace = double.tryParse(_paceDefine) ?? 1.0;

/// Phone-sized on desktop, so screenshots and recordings show the phone
/// layout; the real size on a phone.
const _phone = Size(390, 844);

bool get _desktop => Platform.isMacOS || Platform.isLinux || Platform.isWindows;

final _shotBoundary = GlobalKey();

void _log(String line) {
  // ignore: avoid_print
  print('[demo] $line');
}

// ── Provisioning (Rule 6b: called first in the test body) ──────────────────

Future<void> _ensureModels() async {
  final watch = Stopwatch()..start();
  await bootstrapGemma();

  // EmbeddingGemma 2 (740M): a local copy is accepted only with the pinned
  // bytes, and then recorded as the catalog entry, as an install from the
  // list would be.
  if (_embedderFile.isNotEmpty) {
    final digest = await sha256.bind(File(_embedderFile).openRead()).first;
    expect(
      '$digest',
      _embedder.sha256,
      reason: '$_embedderFile is not the pinned ${_embedder.filename}',
    );
    await FlutterEdgeAi.installEmbedder()
        .modelFromFile(_embedderFile, filename: _embedder.filename)
        .install();
    await persistVerifiedEmbeddingCatalogSelection(
      _embedder,
      spec: EmbeddingModelSpec(
        name: _embedder.filename,
        modelSource: ModelSource.network(_embedder.url),
        modelFilename: _embedder.filename,
      ),
    );
  } else {
    await withCatalogEmbedderSources(
      FlutterEdgeAi.installEmbedder(),
      _embedder,
    ).install();
  }

  // Moonshine Tiny.
  final stt = FlutterEdgeAi.installStt();
  if (_sttModelFile.isNotEmpty) {
    stt.modelFromFile(_sttModelFile).tokenizerFromFile(_sttTokenizerFile);
  } else {
    stt.modelFromNetwork(_stt.modelUrl).tokenizerFromNetwork(_stt.tokenizerUrl);
  }
  await stt.ofType(_stt.sttModelType).install();

  // Gemma 4 E2B.
  final llm = FlutterEdgeAi.installModel(
    modelType: _llm.modelType,
    fileType: _llm.fileType,
  );
  if (_llmFile.isNotEmpty) {
    expect(
      _llmFile.split(Platform.pathSeparator).last,
      _llm.filename,
      reason: 'the catalog recognizes Gemma 4 E2B by its file name',
    );
    llm.fromFile(_llmFile);
  } else {
    llm.fromNetwork(_llm.url);
  }
  await llm.install();

  // A fresh library: no transcripts yet, nothing counted as indexed — the
  // run shows every memo being transcribed and every item embedded.
  await (await SharedPreferences.getInstance()).remove(kKnowledgeLibraryKey);
  _log('models ready in ${watch.elapsedMilliseconds} ms');
}

// ── Driving ────────────────────────────────────────────────────────────────

Future<void> _pause(WidgetTester tester, int ms) =>
    tester.pump(Duration(milliseconds: (ms * _pace).round()));

/// Pumps until [done], or fails after [timeout].
Future<Duration> _waitUntil(
  WidgetTester tester,
  bool Function() done, {
  required String what,
  Duration timeout = const Duration(minutes: 3),
}) async {
  final watch = Stopwatch()..start();
  while (!done()) {
    if (watch.elapsed > timeout) fail('timed out waiting for $what');
    await tester.pump(const Duration(milliseconds: 100));
  }
  return watch.elapsed;
}

/// Types [text] into [field] a character at a time.
Future<void> _type(WidgetTester tester, Finder field, String text) async {
  await tester.tap(field);
  await _pause(tester, 300);
  for (var i = 1; i <= text.length; i++) {
    await tester.enterText(field, text.substring(0, i));
    await tester.pump(Duration(milliseconds: (55 * _pace).round()));
  }
}

String _textOf(WidgetTester tester, Finder finder) {
  final texts = find.descendant(of: finder, matching: find.byType(Text));
  return [
    for (final element in texts.evaluate())
      (element.widget as Text).data ??
          (element.widget as Text).textSpan?.toPlainText() ??
          '',
  ].join(' | ');
}

int _shot = 0;

Future<void> _screenshot(WidgetTester tester, String name) async {
  if (!_shots) return;
  final boundary =
      _shotBoundary.currentContext!.findRenderObject()!
          as RenderRepaintBoundary;
  final image = await boundary.toImage(pixelRatio: 3);
  final png = await image.toByteData(format: ui.ImageByteFormat.png);
  image.dispose();
  final directory = await Directory(
    '${(await getApplicationDocumentsDirectory()).path}/knowledge_shots',
  ).create(recursive: true);
  final file = File(
    '${directory.path}/${(++_shot).toString().padLeft(2, '0')}_$name.png',
  );
  await file.writeAsBytes(png!.buffer.asUint8List());
  _log('screenshot ${file.path}');
}

void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  // Every frame the app asks for, so animations play between pumps.
  binding.framePolicy = LiveTestWidgetsFlutterBindingFramePolicy.fullyLive;

  testWidgets('Knowledge demo: index, find a memo and a photo by text, ask', (
    tester,
  ) async {
    await _ensureModels();
    if (_desktop) await binding.setSurfaceSize(_phone);
    await tester.pumpWidget(
      RepaintBoundary(
        key: _shotBoundary,
        child: const ChatApp(debugBanner: false),
      ),
    );
    final run = Stopwatch()..start();
    await _pause(tester, 1200);

    // Home → Knowledge.
    final card = find.byKey(const Key('home-knowledge'));
    await tester.scrollUntilVisible(
      card,
      300,
      scrollable: find.byType(Scrollable).first,
      duration: const Duration(milliseconds: 400),
    );
    await _pause(tester, 500);
    await tester.tap(card);
    final opened = await _waitUntil(
      tester,
      () => find.byKey(const Key('knowledge-index-card')).evaluate().isNotEmpty,
      what: 'the indexes to open',
    );
    _log('screen ready ${opened.inMilliseconds} ms after the tap');
    await _pause(tester, 1500);
    await _screenshot(tester, 'library');

    // Index the library: notes, photos, memos (transcribed first).
    final indexButton = find.byKey(const Key('knowledge-index'));
    expect(indexButton, findsOneWidget, reason: 'a fresh library');
    await tester.tap(indexButton);
    await _pause(tester, 1800);
    final library = find.byKey(const Key('knowledge-library'));
    await tester.timedDrag(
      library,
      const Offset(0, -420),
      const Duration(milliseconds: 900),
    );
    await _pause(tester, 1500);
    await tester.timedDrag(
      library,
      const Offset(0, -520),
      const Duration(milliseconds: 900),
    );
    final indexed = await _waitUntil(
      tester,
      () => find.byKey(const Key('knowledge-pending')).evaluate().isEmpty,
      what: 'indexing (an item that failed stays pending)',
    );
    _log('indexed ${indexed.inMilliseconds} ms after the scroll');
    await _pause(tester, 1200);
    await _screenshot(tester, 'indexed');
    await tester.timedDrag(
      library,
      const Offset(0, 1400),
      const Duration(milliseconds: 700),
    );
    await _pause(tester, 600);
    expect(
      find.textContaining(RegExp(r'^(\d+) of \1 indexed$')),
      findsOneWidget,
      reason: 'every item indexed',
    );

    // Search: a voice memo, found by what is SAID in it.
    await tester.tap(find.byKey(const Key('knowledge-tab-search')));
    await _pause(tester, 900);
    final field = find.descendant(
      of: find.byKey(const Key('knowledge-search-field')),
      matching: find.byType(EditableText),
    );
    await _type(tester, field, kKnowledgeSuggestions[0]);
    await _pause(tester, 300);
    await tester.tap(find.byKey(const Key('knowledge-search-go')));
    await _waitUntil(
      tester,
      () => find.byKey(const Key('knowledge-result-0')).evaluate().isNotEmpty,
      what: 'the first search',
    );
    await _pause(tester, 1400);
    final first = _textOf(tester, find.byKey(const Key('knowledge-result-0')));
    _log('"${kKnowledgeSuggestions[0]}" → $first');
    expect(first, contains('VOICE MEMO'), reason: 'a memo ranks first');
    await _screenshot(tester, 'search_memo');
    await tester.tap(
      find.descendant(
        of: find.byKey(const Key('knowledge-result-0')),
        matching: find.byKey(const Key('knowledge-play-memo-flight')),
      ),
    );
    await _pause(tester, 5200); // the memo plays (4.5 s)

    // A photo, found by what is IN it.
    await tester.tap(find.byKey(const Key('knowledge-suggestion-1')));
    await _waitUntil(
      tester,
      () => _textOf(
        tester,
        find.byKey(const Key('knowledge-result-0')),
      ).contains('PHOTO'),
      what: 'the photo search',
    );
    await _pause(tester, 3200);
    _log(
      '"${kKnowledgeSuggestions[1]}" → '
      '${_textOf(tester, find.byKey(const Key('knowledge-result-0')))}',
    );
    await _screenshot(tester, 'search_photo');

    // Ask: retrieval across the three kinds, then Gemma 4 with the photo.
    await tester.tap(find.byKey(const Key('knowledge-tab-ask')));
    await _pause(tester, 900);
    final ask = find.descendant(
      of: find.byKey(const Key('knowledge-ask-field')),
      matching: find.byType(EditableText),
    );
    await _type(tester, ask, kKnowledgeQuestions[0]);
    await _pause(tester, 400);
    await tester.tap(find.byKey(const Key('knowledge-ask-send')));
    final answered = await _waitUntil(
      tester,
      () =>
          find.byKey(const Key('knowledge-ask-send')).evaluate().isNotEmpty &&
          find.textContaining('done in').evaluate().isNotEmpty,
      what: 'the answer',
      timeout: const Duration(minutes: 6),
    );
    final answerText = find.byKey(const Key('knowledge-answer-text'));
    final answer = tester
        .widget<Text>(answerText)
        .textSpan!
        .toPlainText(includePlaceholders: false);
    final citations = find.descendant(
      of: answerText,
      matching: find.byWidgetPredicate(
        (w) => w is Text && RegExp(r'^\d+$').hasMatch(w.data ?? ''),
      ),
    );
    _log(
      'answered in ${answered.inMilliseconds} ms, '
      '${citations.evaluate().length} citations: $answer',
    );
    expect(citations, findsWidgets, reason: 'the answer cites its sources');
    await _pause(tester, 4500);
    await _screenshot(tester, 'answer');
    _log('demo ran ${run.elapsed.inSeconds} s');
  });
}
