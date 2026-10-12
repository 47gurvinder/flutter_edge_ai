// The Knowledge screen and its view model over fakes (test/knowledge_fakes.dart)
// with the real flutter_edge_ai_rag in between: indexing the three kinds
// (memos transcribed first), text search ranked across kinds, Ask with
// numbered sources and the photo attached, the model lists it links to, and
// the failures it must show. The real models run in
// integration_test/knowledge_demo_test.dart.

import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_edge_ai_example/knowledge/knowledge_dependencies.dart';
import 'package:flutter_edge_ai_example/knowledge/knowledge_item.dart';
import 'package:flutter_edge_ai_example/knowledge/knowledge_library.dart';
import 'package:flutter_edge_ai_example/knowledge/knowledge_seed.dart';
import 'package:flutter_edge_ai_example/knowledge/knowledge_view_model.dart';
import 'package:flutter_edge_ai_example/knowledge/knowledge_widgets.dart';
import 'package:flutter_edge_ai_example/knowledge_screen.dart';
import 'package:flutter_edge_ai_rag/flutter_edge_ai_rag.dart';
import 'package:flutter_test/flutter_test.dart';

import 'knowledge_fakes.dart';

final _all = kSeedNotes.length + kSeedPhotos.length + kSeedMemos.length;

void main() {
  late KnowledgeRig rig;
  late List<KnowledgeModelList> opened;

  setUp(() {
    rig = KnowledgeRig();
    opened = [];
  });

  Future<void> pumpScreen(WidgetTester tester) async {
    // A phone's width, tall enough that every list is built (they are lazy).
    tester.view
      ..physicalSize = const Size(390, 5200)
      ..devicePixelRatio = 1;
    addTearDown(tester.view.reset);
    await tester.pumpWidget(
      MaterialApp(
        theme: ThemeData(brightness: Brightness.dark),
        home: KnowledgeScreen(
          dependencies: rig.dependencies(),
          openModelList: (_, list) async => opened.add(list),
        ),
      ),
    );
    await settle(tester);
  }

  Future<void> index(WidgetTester tester) async {
    await tester.tap(find.byKey(const Key('knowledge-index')));
    await settle(tester, frames: 80);
  }

  Future<void> tapChip(WidgetTester tester, String key) async {
    await tester.ensureVisible(find.byKey(Key(key)));
    await tester.pump();
    await tester.tap(find.byKey(Key(key)));
    await settle(tester);
  }

  Future<void> tab(WidgetTester tester, String name) async {
    await tester.tap(find.byKey(Key('knowledge-tab-$name')));
    await settle(tester);
  }

  String textIn(WidgetTester tester, Finder finder) => [
    for (final e
        in find.descendant(of: finder, matching: find.byType(Text)).evaluate())
      (e.widget as Text).data ?? (e.widget as Text).textSpan!.toPlainText(),
  ].join(' | ');

  testWidgets('opens on Library with every sample pending, and indexes notes, '
      'photos and memos — each memo transcribed first', (tester) async {
    await pumpScreen(tester);
    expect(find.text('0 of $_all indexed'), findsOneWidget);
    expect(find.byKey(const Key('knowledge-pending')), findsOneWidget);

    await index(tester);

    expect(find.text('$_all of $_all indexed'), findsOneWidget);
    expect(find.byKey(const Key('knowledge-pending')), findsNothing);
    expect(rig.stored(KnowledgeKind.note).keys, {
      for (final n in kSeedNotes) n.id,
    });
    expect(rig.stored(KnowledgeKind.photo).keys, {
      for (final p in kSeedPhotos) p.id,
    });
    expect(rig.stored(KnowledgeKind.memo).keys, {
      for (final m in kSeedMemos) m.id,
    });
    // The index keeps references for media, the text for notes.
    expect(
      rig.stored(KnowledgeKind.photo)['photo-0412']!.$1,
      'asset:assets/eg2_vision/taxi_receipt.png',
    );
    expect(rig.stored(KnowledgeKind.note)['note-pasta']!.$1, startsWith('Spa'));
    // Every memo was transcribed once, and its transcript shows.
    expect(rig.transcribed, [for (final m in kSeedMemos) m.id]);
    expect(find.text('“${kTranscripts['memo-flight']}”'), findsOneWidget);
    // The sound is embedded, not the transcript: memo vectors came from audio.
    expect(rig.embedder.embedded, containsAll(['memo-flight', 'photo-0412']));
    // Saved: the transcripts and what is indexed, for the next launch.
    final saved = KnowledgeLibrary.decode(rig.library.saved);
    expect(
      (saved.byId('memo-flight')! as MemoItem).transcript,
      kTranscripts['memo-flight'],
    );
    expect(saved.indexedIn(kTestTextProfile), hasLength(_all));
    // The language model loads once nothing is waiting to be indexed.
    expect(rig.answerer.warmUps, 1);
    expect(find.textContaining('cpu · loaded in'), findsOneWidget);
  });

  testWidgets('a text query finds a voice memo by its sound and a photo by '
      'its pixels, ranked across kinds, with the search time', (tester) async {
    await pumpScreen(tester);
    await index(tester);
    await tab(tester, 'search');

    await tapChip(tester, 'knowledge-suggestion-0');
    // "my flight is delayed": the memo and the trip note share the topic.
    final first = textIn(tester, find.byKey(const Key('knowledge-result-0')));
    final second = textIn(tester, find.byKey(const Key('knowledge-result-1')));
    expect(
      '$first $second',
      allOf(contains('Voice memo 2'), contains('London')),
    );
    expect(find.byKey(const Key('knowledge-result-2')), findsNothing);
    expect(
      find.textContaining(RegExp(r'^2 RESULTS · \d+ ms$')),
      findsOneWidget,
    );

    await tapChip(tester, 'knowledge-suggestion-1');
    final photo = textIn(tester, find.byKey(const Key('knowledge-result-0')));
    expect(photo, allOf(contains('PHOTO'), contains('IMG_0412')));
    expect(find.byKey(const Key('knowledge-result-1')), findsNothing);

    // A text query over media is sent raw; over notes, as a retrieval query.
    expect(rig.embedder.tasks, contains(TaskType.retrievalQuery));
    expect(rig.embedder.tasks, contains(TaskType.none));
  });

  testWidgets('a memo plays from a result and stops when tapped again', (
    tester,
  ) async {
    await pumpScreen(tester);
    await index(tester);
    await tab(tester, 'search');
    await tester.enterText(
      find.descendant(
        of: find.byKey(const Key('knowledge-search-field')),
        matching: find.byType(EditableText),
      ),
      'the dentist',
    );
    await tester.tap(find.byKey(const Key('knowledge-search-go')));
    await settle(tester);
    final play = find.descendant(
      of: find.byKey(const Key('knowledge-result-0')),
      matching: find.byKey(const Key('knowledge-play-memo-dentist')),
    );
    await tester.tap(play);
    await settle(tester, frames: 5);
    expect(rig.player.played, ['asset:assets/knowledge/memos/memo_6.wav']);
    await tester.tap(play);
    await settle(tester, frames: 5);
    expect(rig.player.played, hasLength(1), reason: 'stopped, not replayed');
  });

  testWidgets('Ask retrieves across kinds, numbers the sources, sends the '
      'best photo, streams the answer and draws its citations', (tester) async {
    await pumpScreen(tester);
    await index(tester);
    await tab(tester, 'ask');

    await tester.tap(find.byKey(const Key('knowledge-question-0')));
    await settle(tester, frames: 60);

    expect(rig.answerer.prompts, hasLength(1));
    final prompt = rig.answerer.prompts.single;
    expect(prompt, contains('[1] Photo: the attached image'));
    expect(prompt, contains('Question: ${kKnowledgeQuestions[0]}'));
    expect(rig.answerer.systems.single, contains('square brackets'));
    expect(utf8.decode(rig.answerer.images.single!), 'photo-0412');
    expect(find.byKey(const Key('knowledge-source-1')), findsOneWidget);
    // [1] and [2] are drawn as pills, not as bracketed text.
    final answer = find.byKey(const Key('knowledge-answer-text'));
    expect(
      find.descendant(of: answer, matching: find.text('1')),
      findsOneWidget,
    );
    expect(
      find.descendant(of: answer, matching: find.text('2')),
      findsOneWidget,
    );
    expect(
      tester.widget<Text>(answer).textSpan!.toPlainText(),
      isNot(contains('[1]')),
    );
    expect(find.textContaining('done in'), findsOneWidget);
    expect(find.textContaining('1 photo seen'), findsOneWidget);
    expect(rig.beforeAnswer, 1);
  });

  testWidgets('a question can be stopped while it streams', (tester) async {
    await pumpScreen(tester);
    await index(tester);
    await tab(tester, 'ask');
    rig.answerer.gate = Completer<void>();
    await tester.tap(find.byKey(const Key('knowledge-question-1')));
    await settle(tester, frames: 10);
    expect(find.byKey(const Key('knowledge-ask-stop')), findsOneWidget);
    await tester.tap(find.byKey(const Key('knowledge-ask-stop')));
    rig.answerer.gate!.complete();
    await settle(tester);
    expect(rig.answerer.stopped, isTrue);
    expect(find.byKey(const Key('knowledge-ask-send')), findsOneWidget);
  });

  testWidgets('no embedder: says what to install and opens the Embedding '
      'Models list, then opens again', (tester) async {
    rig.withEmbedder = false;
    await pumpScreen(tester);
    expect(find.textContaining('No embedding model is active'), findsOneWidget);
    expect(find.byType(NavigationBar), findsNothing);

    rig.withEmbedder = true;
    await tester.tap(find.byKey(const Key('knowledge-open-embedder-list')));
    await settle(tester);
    expect(opened, [KnowledgeModelList.embedder]);
    expect(find.byKey(const Key('knowledge-index')), findsOneWidget);
  });

  testWidgets('without the 740M memos are listed but not indexed, with the '
      'way to the list; without speech-to-text memos have no transcript', (
    tester,
  ) async {
    rig = KnowledgeRig(embedder: TopicEmbedder(audio: false))..withStt = false;
    await pumpScreen(tester);
    final memosOnly = kSeedNotes.length + kSeedPhotos.length;
    expect(find.text('0 of $memosOnly indexed'), findsOneWidget);
    expect(
      find.textContaining('Voice memos need EmbeddingGemma 2'),
      findsOneWidget,
    );
    expect(find.textContaining('Install a speech-to-text'), findsOneWidget);
    await index(tester);
    expect(find.text('$memosOnly of $memosOnly indexed'), findsOneWidget);
    expect(rig.stored(KnowledgeKind.memo), isEmpty);
    expect(rig.transcribed, isEmpty);
  });

  testWidgets('without a language model Ask says what to install', (
    tester,
  ) async {
    rig.withLlm = false;
    await pumpScreen(tester);
    await tab(tester, 'ask');
    expect(
      find.textContaining('Answers need a language model'),
      findsOneWidget,
    );
    await tester.tap(find.text('Install'));
    await settle(tester);
    expect(opened, [KnowledgeModelList.llm]);
    expect(rig.answerer.warmUps, 0);
  });

  testWidgets('a photo the platform cannot decode fails alone; the rest are '
      'indexed', (tester) async {
    rig.embedder.failures['photo-0455'] =
        const EmbeddingFormatNotSupportedException(
          'LiteRT-LM does not decode heic images.',
          modality: EmbeddingModality.image,
          detected: null,
          accepted: {},
        );
    await pumpScreen(tester);
    await index(tester);
    expect(find.text('${_all - 1} of $_all indexed'), findsOneWidget);
    expect(rig.stored(KnowledgeKind.photo).keys, isNot(contains('photo-0455')));
    expect(
      find.descendant(
        of: find.byKey(const Key('knowledge-item-photo-0455')),
        matching: find.byIcon(Icons.error_outline),
      ),
      findsOneWidget,
    );
    expect(find.byKey(const Key('knowledge-pending')), findsOneWidget);
  });

  testWidgets('saved index state the index contradicts is dropped, so those '
      'items are indexed again', (tester) async {
    // The library says every note is indexed; the note index is empty.
    final library = KnowledgeLibrary.seeded();
    for (final note in kSeedNotes) {
      library.markIndexed(kTestTextProfile, note.id, (
        embed: const Duration(milliseconds: 5),
        transcribe: null,
      ));
    }
    rig.library.saved = library.encode();
    await pumpScreen(tester);
    expect(find.text('0 of $_all indexed'), findsOneWidget);
  });

  testWidgets('a note, a photo and a recorded memo are added and indexed', (
    tester,
  ) async {
    await pumpScreen(tester);
    await index(tester);

    await tester.tap(find.byKey(const Key('knowledge-add-note')));
    await settle(tester);
    await tester.enterText(
      find.byKey(const Key('knowledge-note-title')),
      'Winter tires',
    );
    await tester.enterText(
      find.byKey(const Key('knowledge-note-body')),
      'Book the tires change.',
    );
    await tester.tap(find.byKey(const Key('knowledge-note-save')));
    await settle(tester);
    expect(rig.stored(KnowledgeKind.note), hasLength(kSeedNotes.length + 1));

    rig.pickedPhoto = (
      name: 'chart.png',
      bytes: Uint8List.fromList(utf8.encode('photo-0291')),
    );
    await tester.tap(find.byKey(const Key('knowledge-add-photo')));
    await settle(tester);
    expect(rig.stored(KnowledgeKind.photo), hasLength(kSeedPhotos.length + 1));
    expect(rig.media.kept.keys.single, startsWith('/kept/photo-'));

    await tester.tap(find.byKey(const Key('knowledge-record')));
    await settle(tester, frames: 5);
    expect(find.byKey(const Key('knowledge-recording')), findsOneWidget);
    await tester.tap(find.byKey(const Key('knowledge-stop-recording')));
    await settle(tester);
    expect(rig.stored(KnowledgeKind.memo), hasLength(kSeedMemos.length + 1));
    expect(find.textContaining('Voice memo 7'), findsOneWidget);
    expect(find.text('${_all + 3} of ${_all + 3} indexed'), findsOneWidget);
  });

  testWidgets('a recorder without the microphone shows why', (tester) async {
    rig.recorder.startError = const KnowledgeRecorderException(
      'The microphone is not available.',
    );
    await pumpScreen(tester);
    await tester.tap(find.byKey(const Key('knowledge-record')));
    await settle(tester);
    expect(find.byKey(const Key('knowledge-error')), findsOneWidget);
    expect(find.byKey(const Key('knowledge-recording')), findsNothing);
  });

  testWidgets('on the web, embedding releases the language model and the '
      'next question loads it again', (tester) async {
    await pumpScreen(tester);
    await index(tester);
    expect(rig.answerer.warmUps, 1);
    rig.releaseLlmOnEmbedding = true;
    await tab(tester, 'search');
    await tapChip(tester, 'knowledge-suggestion-1');
    await tab(tester, 'library');
    expect(find.textContaining('loads when needed'), findsOneWidget);
    await tab(tester, 'ask');
    await tester.tap(find.byKey(const Key('knowledge-question-0')));
    await settle(tester, frames: 60);
    expect(rig.answerer.warmUps, 2);
  });

  test('the view model keeps the samples when nothing was saved', () {
    final vm = KnowledgeViewModel(rig.dependencies());
    expect(vm.items, isNotEmpty);
    vm.dispose();
  });
}
