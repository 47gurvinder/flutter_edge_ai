// The Knowledge screen and its view model over fakes (test/knowledge_fakes.dart)
// with the real flutter_edge_ai_rag in between: indexing the three kinds (a
// memo by its sound, never a transcript), text search ranked across kinds,
// Find similar, Ask with numbered sources and the photo attached, Ask by
// voice, Live (windows surfacing items by sound, a heard question answered,
// the meeting searched), the model lists it links to, and the failures it
// must show. The real models run in integration_test/knowledge_demo_test.dart.

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
import 'package:flutter_edge_ai_example/utils/audio_converter.dart';
import 'package:flutter_edge_ai_rag/flutter_edge_ai_rag.dart';
import 'package:flutter_test/flutter_test.dart';

import 'knowledge_fakes.dart';

final _all = kSeedNotes.length + kSeedPhotos.length + kSeedMemos.length;

void main() {
  late KnowledgeRig rig;
  late List<KnowledgeModelList> opened;

  setUp(() {
    rig = KnowledgeRig()..answerer.reply = 'The total was 26.50 [1].';
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

  Future<void> tapKey(
    WidgetTester tester,
    String key, {
    int frames = 40,
  }) async {
    await tester.ensureVisible(find.byKey(Key(key)));
    await tester.pump();
    await tester.tap(find.byKey(Key(key)));
    await settle(tester, frames: frames);
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

  String? topicOfWav(Uint8List? wav) =>
      wav == null ? null : topicOfPcm(AudioConverter.parseWav(wav).pcmData);

  testWidgets('opens on Library with every sample pending, and indexes notes, '
      'photos and memos — a memo by its sound', (tester) async {
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
      'asset:assets/knowledge/photos/taxi_receipt.png',
    );
    expect(rig.stored(KnowledgeKind.note)['note-pasta']!.$1, startsWith('Spa'));
    // Memo vectors came from their sound; the photo's from its pixels.
    expect(
      rig.embedder.embedded,
      containsAll(['audio:flight', 'audio:dentist', 'photo-0412']),
    );
    // Saved: what is indexed, for the next launch.
    final saved = KnowledgeLibrary.decode(rig.library.saved);
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

    await tapKey(tester, 'knowledge-suggestion-0');
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

    await tapKey(tester, 'knowledge-suggestion-1');
    final photo = textIn(tester, find.byKey(const Key('knowledge-result-0')));
    expect(photo, contains('IMG_0412'));
    expect(find.byKey(const Key('knowledge-result-1')), findsNothing);

    // A text query over media is sent raw; over notes, as a retrieval query.
    expect(rig.embedder.tasks, contains(TaskType.retrievalQuery));
    expect(rig.embedder.tasks, contains(TaskType.none));
  });

  testWidgets('typing searches once the user pauses', (tester) async {
    await pumpScreen(tester);
    await index(tester);
    await tab(tester, 'search');
    final field = find.descendant(
      of: find.byKey(const Key('knowledge-search-field')),
      matching: find.byType(EditableText),
    );
    final before = rig.embedder.embedded.length;
    await tester.enterText(field, 'the den');
    await tester.pump(const Duration(milliseconds: 100));
    await tester.enterText(field, 'the dentist');
    await settle(tester);
    expect(
      rig.embedder.embedded.length - before,
      3,
      reason: 'one search, for the last text: an embedding per index',
    );
    expect(
      textIn(tester, find.byKey(const Key('knowledge-result-0'))),
      contains('Voice memo 6'),
    );
  });

  testWidgets('Find similar searches the photos with a photo', (tester) async {
    await pumpScreen(tester);
    await index(tester);
    await tab(tester, 'search');
    await tester.enterText(
      find.descendant(
        of: find.byKey(const Key('knowledge-search-field')),
        matching: find.byType(EditableText),
      ),
      'market share chart',
    );
    await tester.tap(find.byKey(const Key('knowledge-search-go')));
    await settle(tester);
    expect(find.byKey(const Key('knowledge-similar-photo-0291')), findsOne);

    await tapKey(tester, 'knowledge-similar-photo-0291');
    final first = textIn(tester, find.byKey(const Key('knowledge-result-0')));
    expect(first, contains('IMG_0290'), reason: 'the other chart');
    expect(find.textContaining('image → image'), findsOneWidget);
    expect(
      rig.embedder.embedded.where((e) => e == 'photo-0291'),
      hasLength(2),
      reason: 'indexed once, then the query',
    );
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
    expect(rig.answerer.images.single.map(utf8.decode), ['photo-0412']);
    expect(rig.answerer.audios.single, isNull);
    expect(find.byKey(const Key('knowledge-source-1')), findsOneWidget);
    expect(find.byKey(const Key('knowledge-citation-1')), findsOneWidget);
    // [1] is drawn as a pill, not as bracketed text.
    final answer = find.byKey(const Key('knowledge-answer-text'));
    expect(
      find.descendant(of: answer, matching: find.text('1')),
      findsOneWidget,
    );
    expect(
      tester.widget<Text>(answer).textSpan!.toPlainText(),
      allOf(contains('26.50'), isNot(contains('[1]'))),
    );
    final stats = tester
        .widget<Text>(find.byKey(const Key('knowledge-answer-stats')))
        .data!;
    expect(stats, allOf(contains('retrieved in'), contains('photo sent')));
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

  testWidgets('Ask by voice: the spoken question is the query and Gemma 4 '
      'hears it with the photo — no transcript anywhere', (tester) async {
    await pumpScreen(tester);
    await index(tester);
    await tab(tester, 'ask');

    await tester.tap(find.byKey(const Key('knowledge-ask-voice')));
    await settle(tester, frames: 10);
    await tester.tap(find.byKey(const Key('knowledge-ask-voice-sample')));
    await settle(tester, frames: 10);
    expect(rig.player.played, [kSampleQuestion]);
    expect(find.byKey(const Key('knowledge-ask-voice-bubble')), findsOneWidget);
    expect(rig.answerer.prompts, isEmpty, reason: 'still playing');

    await settle(tester, frames: 120);
    expect(rig.answerer.prompts, hasLength(1));
    final prompt = rig.answerer.prompts.single;
    expect(prompt, contains('it is the attached recording'));
    expect(prompt, isNot(contains('Question:')));
    expect(topicOfWav(rig.answerer.audios.single), 'taxi');
    expect(rig.answerer.images.single.map(utf8.decode), ['photo-0412']);
    expect(find.byKey(const Key('knowledge-citation-1')), findsOneWidget);
    expect(find.text(kSpokenLabel), findsNothing);
    expect(find.byKey(const Key('knowledge-ask-question')), findsNothing);
  });

  testWidgets('a recorded question is asked when recording stops', (
    tester,
  ) async {
    rig.recorder.recording = topicWav('dentist', const Duration(seconds: 1));
    await pumpScreen(tester);
    await index(tester);
    await tab(tester, 'ask');
    await tester.tap(find.byKey(const Key('knowledge-ask-voice')));
    await settle(tester, frames: 10);
    await tester.tap(find.byKey(const Key('knowledge-ask-voice-mic')));
    await settle(tester, frames: 10);
    expect(rig.recorder.started, isTrue);
    await tapKey(tester, 'knowledge-ask-voice-stop', frames: 60);
    expect(topicOfWav(rig.answerer.audios.single), 'dentist');
    // The best source is a memo, but the question is the one recording sent.
    expect(rig.answerer.prompts.single, contains('Voice memo (not played'));
  });

  group('Live', () {
    Future<void> openLive(WidgetTester tester) async {
      await pumpScreen(tester);
      await index(tester);
      await tab(tester, 'live');
    }

    /// Speaks [pcm] into the microphone in 100 ms chunks, in real time.
    Future<void> speak(WidgetTester tester, Uint8List pcm) async {
      for (var i = 0; i < pcm.length; i += 3200) {
        rig.microphone!.add(
          Uint8List.sublistView(pcm, i, (i + 3200).clamp(0, pcm.length)),
        );
        await tester.pump(const Duration(milliseconds: 100));
      }
    }

    Finder surfacedCards() => find.byWidgetPredicate(
      (w) =>
          w.key is ValueKey<String> &&
          (w.key! as ValueKey<String>).value.startsWith('knowledge-surfaced-'),
    );

    testWidgets('each window is matched by sound: what it is about surfaces, '
        'a heard question is answered from the library, and the meeting '
        'can be searched', (tester) async {
      await openLive(tester);
      await tapKey(tester, 'knowledge-live-source-Microphone');
      await tapKey(tester, 'knowledge-live-start', frames: 5);
      expect(rig.microphoneStarts, 1);

      await speak(tester, kMeetingPcm);
      await settle(tester, frames: 80);

      // Small talk surfaces nothing; the dentist, then the taxi, do.
      expect(
        find.byKey(const Key('knowledge-surfaced-memo-dentist')),
        findsOneWidget,
      );
      expect(
        find.byKey(const Key('knowledge-surfaced-photo-0412')),
        findsOneWidget,
      );
      expect(surfacedCards(), findsNWidgets(2));
      // Gemma 4 heard the taxi window ask a question: it is answered.
      expect(
        textIn(tester, find.byKey(const Key('knowledge-live-question'))),
        contains('How much was the airport taxi?'),
      );
      expect(
        textIn(tester, find.byKey(const Key('knowledge-live-answer'))),
        contains('26.50'),
      );
      expect(rig.answerer.images.single.map(utf8.decode), ['photo-0412']);
      expect(
        find.byKey(const Key('knowledge-marker-memo-dentist')),
        findsOneWidget,
      );
      expect(
        find.byKey(const Key('knowledge-marker-question')),
        findsOneWidget,
      );
      expect(find.byKey(const Key('knowledge-live-timings')), findsOneWidget);

      // The microphone stops; the meeting can be searched by meaning.
      await tapKey(tester, 'knowledge-live-stop');
      expect(rig.microphone!.hasListener, isFalse);
      await tapKey(tester, 'knowledge-meeting-search');
      await tester.enterText(
        find.descendant(
          of: find.byKey(const Key('knowledge-meeting-field')),
          matching: find.byType(EditableText),
        ),
        'the dentist',
      );
      await tester.tap(find.byKey(const Key('knowledge-meeting-go')));
      await settle(tester);
      expect(find.byKey(const Key('knowledge-live-found')), findsOneWidget);
      expect(find.textContaining('talked about at'), findsOneWidget);
    });

    testWidgets('"Answer what was just said" sends the last seconds as the '
        'question', (tester) async {
      rig.answerer.hear = (_) => 'NONE';
      await openLive(tester);
      await tapKey(tester, 'knowledge-live-source-Microphone');
      await tapKey(tester, 'knowledge-live-start', frames: 5);
      await speak(tester, kMeetingPcm);
      await settle(tester, frames: 80);
      expect(find.byKey(const Key('knowledge-live-question')), findsNothing);

      await tapKey(tester, 'knowledge-live-menu', frames: 10);
      await tapKey(tester, 'knowledge-live-just-said', frames: 80);
      expect(rig.answerer.prompts.single, contains('attached recording'));
      // Every window of the last ten seconds, as one recording.
      expect(
        AudioConverter.parseWav(rig.answerer.audios.single!).pcmData.length,
        greaterThan(6 * 32000),
      );
      expect(find.byKey(const Key('knowledge-live-answer')), findsOneWidget);
    });

    testWidgets('the sample meeting plays and is heard in real time', (
      tester,
    ) async {
      rig.now = () => tester.binding.clock.now();
      await openLive(tester);
      await tapKey(tester, 'knowledge-live-start', frames: 5);
      expect(rig.player.played, [kSampleMeeting]);
      // Real time: after a second, a second has been heard.
      await settle(tester, frames: 50);
      expect(
        tester.widget<Text>(find.byKey(const Key('knowledge-live-title'))).data,
        '0:01',
      );
      await settle(tester, frames: 400);
      expect(
        find.byKey(const Key('knowledge-surfaced-photo-0412')),
        findsOneWidget,
      );
    });

    testWidgets('a microphone that fails says why', (tester) async {
      await openLive(tester);
      await tapKey(tester, 'knowledge-live-source-Microphone');
      await tapKey(tester, 'knowledge-live-start', frames: 5);
      rig.microphone!.addError(
        const KnowledgeRecorderException('The microphone is not allowed.'),
      );
      await settle(tester);
      expect(find.byKey(const Key('knowledge-error')), findsOneWidget);
      expect(find.byKey(const Key('knowledge-live-start')), findsOneWidget);
    });
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
      'way to the list, and there is no Live', (tester) async {
    rig = KnowledgeRig(embedder: TopicEmbedder(audio: false));
    await pumpScreen(tester);
    final memosOnly = kSeedNotes.length + kSeedPhotos.length;
    expect(find.text('0 of $memosOnly indexed'), findsOneWidget);
    expect(
      find.textContaining('Voice memos need EmbeddingGemma 2'),
      findsOneWidget,
    );
    expect(find.byKey(const Key('knowledge-tab-live')), findsNothing);
    await index(tester);
    expect(find.text('$memosOnly of $memosOnly indexed'), findsOneWidget);
    expect(rig.stored(KnowledgeKind.memo), isEmpty);
  });

  testWidgets('without a language model Ask says what to install', (
    tester,
  ) async {
    rig.withLlm = false;
    await pumpScreen(tester);
    expect(find.byKey(const Key('knowledge-tab-live')), findsNothing);
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
      library.markIndexed(
        kTestTextProfile,
        note.id,
        const Duration(milliseconds: 5),
      );
    }
    rig.library.saved = library.encode();
    await pumpScreen(tester);
    expect(find.text('0 of $_all indexed'), findsOneWidget);
  });

  testWidgets('documents the library does not know are cleared from the '
      'index before it is filled again', (tester) async {
    await pumpScreen(tester);
    await index(tester);
    // Another run left a document behind and its save was lost.
    rig.stored(KnowledgeKind.note)['note-gone'] = (
      'gone',
      rig.stored(KnowledgeKind.note)['note-q3']!.$2,
    );
    rig.library.saved = null;
    await tester.pumpWidget(const SizedBox());
    await pumpScreen(tester);
    expect(rig.stored(KnowledgeKind.note), isEmpty);
    await index(tester);
    expect(rig.stored(KnowledgeKind.note).keys, isNot(contains('note-gone')));
    expect(find.text('$_all of $_all indexed'), findsOneWidget);
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
    await tapKey(tester, 'knowledge-suggestion-1');
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
