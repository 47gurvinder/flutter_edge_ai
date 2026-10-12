// The Knowledge screen's pure parts: ranking across kinds by lift, snippet
// highlighting, splitting a memo for a speech-to-text window, the prompt and
// its citations, and the saved library.

import 'dart:typed_data';

import 'package:flutter_edge_ai_example/knowledge/knowledge_audio.dart';
import 'package:flutter_edge_ai_example/knowledge/knowledge_item.dart';
import 'package:flutter_edge_ai_example/knowledge/knowledge_library.dart';
import 'package:flutter_edge_ai_example/knowledge/knowledge_prompt.dart';
import 'package:flutter_edge_ai_example/knowledge/knowledge_ranking.dart';
import 'package:flutter_edge_ai_example/knowledge/knowledge_seed.dart';
import 'package:flutter_edge_ai_example/utils/audio_converter.dart';
import 'package:flutter_test/flutter_test.dart';

List<ScoredDocument> _docs(String prefix, List<double> scores) => [
  for (final (i, s) in scores.indexed) (id: '$prefix$i', similarity: s),
];

void main() {
  group('rankAcrossKinds', () {
    test('ranks by lift over each kind\'s median, not by raw cosine', () {
      // Measured shape: memos sit higher across the board than notes.
      final ranked = rankAcrossKinds({
        KnowledgeKind.note: _docs('n', [0.52, 0.55, 0.56, 0.57, 0.77]),
        KnowledgeKind.memo: _docs('m', [0.62, 0.64, 0.66, 0.68, 0.72]),
      });
      // n4 lifts 0.21; m4 has the higher cosine but lifts only 0.06.
      expect(ranked.map((h) => h.id), ['n4', 'm4']);
      expect(ranked.first.lift, closeTo(0.21, 1e-9));
      expect(ranked.first.strength, 1.0);
      expect(ranked.last.strength, closeTo(0.06 / kFullLift, 1e-9));
    });

    test('leaves out what does not stand out, and keeps at most limit', () {
      final flat = rankAcrossKinds({
        KnowledgeKind.photo: _docs('p', [0.58, 0.59, 0.6, 0.61]),
      });
      expect(flat, isEmpty);
      final many = rankAcrossKinds({
        KnowledgeKind.note: _docs('n', [0.5, 0.5, 0.5, 0.9, 0.9, 0.9]),
      }, limit: 2);
      expect(many, hasLength(2));
    });

    test('a kind with too few documents is measured against its measured '
        'baseline', () {
      final ranked = rankAcrossKinds({
        KnowledgeKind.memo: _docs('m', [0.72, 0.66]),
      });
      expect(ranked.single.id, 'm0');
      expect(
        ranked.single.lift,
        closeTo(0.72 - kFallbackBaseline[KnowledgeKind.memo]!, 1e-9),
      );
    });
  });

  group('highlight', () {
    String marked(String text, String query) => [
      for (final part in highlight(text, query))
        if (part.match) part.text,
    ].join(',');

    test('marks the query\'s words by stem, not stop words', () {
      expect(
        marked('My flight to London is delayed.', 'my flight is delays'),
        'flight,delayed',
      );
      expect(marked('Call Mom on Sunday', 'what did I say'), '');
    });

    test('keeps every character, in order', () {
      const text = 'The kitchen lights keep flickering.';
      expect(
        highlight(text, 'flickering lights').map((p) => p.text).join(),
        text,
      );
    });
  });

  group('splitForSpeechWindow', () {
    Uint8List speech(List<(int ms, int amplitude)> spans) {
      final samples = <int>[];
      for (final (ms, amplitude) in spans) {
        for (var i = 0; i < ms * 16; i++) {
          samples.add(i.isEven ? amplitude : -amplitude);
        }
      }
      final bytes = ByteData(samples.length * 2);
      for (final (i, s) in samples.indexed) {
        bytes.setInt16(i * 2, s, Endian.little);
      }
      return bytes.buffer.asUint8List();
    }

    test('a memo inside the window stays whole', () {
      final pcm = speech([(4000, 1000)]);
      expect(splitForSpeechWindow(pcm, window: const Duration(seconds: 5)), [
        pcm,
      ]);
    });

    test('a longer memo is cut at its quietest moment before each window '
        'ends, losing nothing', () {
      // 4.2 s of speech, a 300 ms pause, then 4 s more.
      final pcm = speech([(4200, 3000), (300, 0), (4000, 3000)]);
      final pieces = splitForSpeechWindow(
        pcm,
        window: const Duration(seconds: 5),
      );
      expect(pieces, hasLength(2));
      expect(pieces.fold<int>(0, (n, p) => n + p.length), pcm.length);
      final firstMs = pieces.first.length ~/ 32;
      expect(firstMs, inInclusiveRange(4200, 4500), reason: 'in the pause');
      for (final piece in pieces) {
        expect(piece.length, lessThanOrEqualTo(5 * kPcmBytesPerSecond));
        expect(piece.length.isEven, isTrue);
      }
    });

    test('reads a WAV at another rate as 16 kHz mono', () {
      final wav = AudioConverter.pcmToWav(
        Uint8List(44100 * 2),
        sampleRate: 44100,
      );
      expect(
        pcmDuration(pcm16kMonoFromWav(wav)).inMilliseconds,
        closeTo(1000, 2),
      );
    });
  });

  group('prompt', () {
    const note = NoteItem(id: 'n', title: 'London trip', body: 'Terminal 5.');
    const photo = PhotoItem(id: 'p', title: 'IMG_1', reference: 'asset:x');
    const memo = MemoItem(
      id: 'm',
      title: 'Voice memo 1',
      reference: 'asset:y',
      duration: Duration(seconds: 4),
      transcript: ' Call Mom. ',
    );

    test('numbers the sources and says which photo is attached', () {
      final prompt = buildKnowledgePrompt(' When? ', [
        (number: 1, item: photo, attached: true),
        (number: 2, item: note, attached: false),
        (number: 3, item: memo, attached: false),
        (number: 4, item: photo, attached: false),
      ]);
      expect(
        prompt,
        'Sources:\n'
        '[1] Photo: the attached image\n'
        '[2] Note "London trip": Terminal 5.\n'
        '[3] Voice memo, transcript: "Call Mom."\n'
        '[4] Photo (not shown to you)\n'
        '\n'
        'Question: When?',
      );
    });

    test('reads the citations of an answer', () {
      expect(citedSources('It was 26.50 [1]. Covered [2, 3][1].'), [1, 2, 3]);
      expect(citedSources('No sources.'), isEmpty);
    });
  });

  group('library', () {
    test('saves the user\'s items, the samples\' transcripts and what is '
        'indexed per embedding space', () {
      final library = KnowledgeLibrary.seeded()
        ..add(const NoteItem(id: 'note-1', title: 'Mine', body: 'text'))
        ..replace(
          (KnowledgeLibrary.seeded().byId('memo-mom')! as MemoItem)
              .withTranscript('Call Mom on Sunday.'),
        )
        ..markIndexed('space-a', 'note-1', (
          embed: const Duration(milliseconds: 40),
          transcribe: null,
        ))
        ..markIndexed('space-a', 'memo-mom', (
          embed: const Duration(milliseconds: 160),
          transcribe: const Duration(milliseconds: 90),
        ));
      final restored = KnowledgeLibrary.decode(library.encode());
      expect(restored.items, hasLength(library.items.length));
      expect((restored.byId('note-1')! as NoteItem).body, 'text');
      expect(
        (restored.byId('memo-mom')! as MemoItem).transcript,
        'Call Mom on Sunday.',
      );
      expect(
        restored.indexedIn('space-a')['memo-mom']!.transcribe,
        const Duration(milliseconds: 90),
      );
      expect(restored.indexedIn('space-b'), isEmpty);
      restored.forgetIndexed('space-a', kind: KnowledgeKind.memo);
      expect(restored.indexedIn('space-a').keys, ['note-1']);
    });

    test('starts with every sample', () {
      expect(
        KnowledgeLibrary.seeded().items,
        hasLength(kSeedNotes.length + kSeedPhotos.length + kSeedMemos.length),
      );
    });

    test('a corrupt save is reported', () {
      expect(
        () => KnowledgeLibrary.decode('[1, 2]'),
        throwsA(isA<FormatException>()),
      );
    });
  });
}
