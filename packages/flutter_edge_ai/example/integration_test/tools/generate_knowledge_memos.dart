// The Knowledge screen's sample voice memos, spoken by OUR on-device TTS:
// flutter_edge_ai_speech's Matcha-TTS (litert-community/Matcha-TTS, MIT; its
// voice is LJSpeech, a public-domain recording). No recorded person and no
// macOS `say` voice (licensed for personal, non-commercial use only).
//
// Step 1 — synthesize on macOS (writes <documents>/knowledge_memos/*.wav at
// Matcha's own rate, 22.05 kHz; on macOS <documents> is
// ~/Library/Containers/dev.flutterberlin.flutterGemmaExample55/Data/Documents):
//
//   cd packages/flutter_edge_ai/example
//   flutter test integration_test/tools/generate_knowledge_memos.dart -d macos
//
// Step 2 — resample to 16 kHz mono PCM16, the rate the STT models and the
// EmbeddingGemma 2 audio encoder work at, into assets/knowledge/memos/:
//
//   python3 -I integration_test/tools/convert_knowledge_memos.py \
//     ~/Library/Containers/dev.flutterberlin.flutterGemmaExample55/Data/Documents/knowledge_memos
//
// Installs Matcha from Hugging Face when it is not installed (a no-op when it
// is). Each memo is kept under five seconds: Moonshine, the smallest STT model,
// hears five-second windows. Matcha samples noise, so a rerun writes different
// bytes; the committed clips are the ones the screen was measured with.

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show debugPrint;
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_example/knowledge/knowledge_seed.dart';
import 'package:flutter_edge_ai_example/models/tts_model.dart';
import 'package:flutter_edge_ai_example/utils/audio_converter.dart';
import 'package:flutter_edge_ai_speech/flutter_edge_ai_speech.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:path_provider/path_provider.dart';

/// What each memo says, keyed by the memo id in [kSeedMemos].
const memoScripts = <String, String>{
  'memo-kitchen':
      'The kitchen lights keep flickering. '
      'Please call the electrician on Thursday.',
  'memo-flight':
      'My flight to London is delayed by three hours, tell the hotel.',
  'memo-groceries': 'Pick up milk, fresh basil and lemons after work.',
  'memo-offsite':
      'Idea for the team retreat: a cooking class instead of bowling.',
  'memo-mom': 'Call Mom on Sunday to plan her birthday dinner.',
  'memo-dentist': 'The dentist moved my check-up to Tuesday at noon.',
};

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('synthesizes the sample voice memos with Matcha', (tester) async {
    expect(memoScripts.keys.toSet(), {
      for (final memo in kSeedMemos) memo.id,
    }, reason: 'one script per seeded memo');
    await FlutterEdgeAi.initialize(ttsBackends: const [LiteRtTtsBackend()]);
    await FlutterEdgeAi.installTts()
        .fromNetwork(TtsModel.matcha.baseUrl)
        .ofType(TtsModel.matcha.ttsModelType)
        .install();
    final synth = await FlutterEdgeAi.getActiveTts();
    try {
      final out = Directory(
        '${(await getApplicationDocumentsDirectory()).path}/knowledge_memos',
      );
      await out.create(recursive: true);
      final tooLong = <String>[];
      for (final memo in kSeedMemos) {
        final script = memoScripts[memo.id]!;
        final watch = Stopwatch()..start();
        final pcm = await synth.synthesize(script);
        watch.stop();
        final seconds = pcm.length / 2 / synth.sampleRate;
        final file = File('${out.path}/${memo.assetName}');
        await file.writeAsBytes(
          AudioConverter.pcmToWav(
            Uint8List.fromList(pcm),
            sampleRate: synth.sampleRate,
          ),
          flush: true,
        );
        debugPrint(
          '[memos] ${memo.assetName}: ${seconds.toStringAsFixed(2)} s at '
          '${synth.sampleRate} Hz, synthesized in '
          '${watch.elapsedMilliseconds} ms — "$script"',
        );
        if (seconds >= 5.0) tooLong.add('${memo.id}: $seconds s');
      }
      expect(tooLong, isEmpty, reason: 'each memo must fit a 5 s STT window');
      debugPrint('[memos] wrote ${kSeedMemos.length} clips to ${out.path}');
    } finally {
      await synth.close();
    }
  });
}
