// The pure pieces both embedding hosts share (R5 §4): RequestLanes and the
// pump (W1-W3), checkLoadedPass (§4.2 step 5), and the failure wire (§4.7).

import 'dart:async';

import 'package:flutter_edge_ai/src/embedding/embedding_exceptions.dart';
import 'package:flutter_edge_ai/src/embedding/embedding_modality.dart';
import 'package:flutter_edge_ai/src/embedding/embedding_pass.dart';
import 'package:flutter_edge_ai/src/embedding/load_check.dart';
import 'package:flutter_edge_ai/src/embedding/request_lanes.dart';
import 'package:flutter_edge_ai/src/embedding/wire.dart';
import 'package:flutter_edge_ai/src/media/encoded_media_format.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('RequestLanes', () {
    test('oldest text first, then oldest media; FIFO within a lane', () {
      final lanes = RequestLanes<String>()
        ..add('m1', media: true)
        ..add('t1', media: false)
        ..add('m2', media: true)
        ..add('t2', media: false);

      expect(lanes.length, 4);
      expect(
        [for (var r = lanes.takeNext(); r != null; r = lanes.takeNext()) r],
        ['t1', 't2', 'm1', 'm2'],
      );
      expect(lanes.isEmpty, isTrue);
    });

    test('a text request filed later still overtakes queued media', () {
      final lanes = RequestLanes<String>()
        ..add('m1', media: true)
        ..add('m2', media: true);
      expect(lanes.takeNext(), 'm1');
      lanes.add('t1', media: false);
      expect(lanes.takeNext(), 't1');
      expect(lanes.takeNext(), 'm2');
    });

    test('drain empties both lanes', () {
      final lanes = RequestLanes<String>()
        ..add('m1', media: true)
        ..add('t1', media: false);
      expect(lanes.drain(), unorderedEquals(['t1', 'm1']));
      expect(lanes.isEmpty, isTrue);
    });
  });

  group('RequestPump', () {
    test('serves one request at a time, text lane first', () async {
      final served = <String>[];
      final gate = Completer<void>();
      var inFlight = 0;
      var maxInFlight = 0;
      final pump = RequestPump<String>((r) async {
        inFlight++;
        if (inFlight > maxInFlight) maxInFlight = inFlight;
        served.add(r);
        if (r == 'first') await gate.future;
        inFlight--;
      });
      final done = pump.run();

      pump.file('first', media: false);
      await pumpEventQueue();
      pump
        ..file('m1', media: true)
        ..file('t1', media: false)
        ..file('m2', media: true)
        ..file('t2', media: false);
      gate.complete();
      await pumpEventQueue();
      pump.requestClose();
      await done;

      expect(served, ['first', 't1', 't2', 'm1', 'm2']);
      expect(maxInFlight, 1);
    });

    test(
      'close returns what has not started; the one in flight finishes',
      () async {
        final gate = Completer<void>();
        final finished = <String>[];
        final pump = RequestPump<String>((r) async {
          await gate.future;
          finished.add(r);
        });
        final done = pump.run();
        pump.file('a', media: false);
        await pumpEventQueue();
        pump
          ..file('b', media: false)
          ..file('c', media: true);

        final notStarted = pump.requestClose();
        expect(notStarted, ['b', 'c']);
        expect(pump.file('d', media: false), isFalse);

        gate.complete();
        await done;
        expect(finished, ['a']);
      },
    );

    test('a request filed after close is refused, not queued', () {
      final pump = RequestPump<String>((_) async {})..requestClose();
      expect(pump.file('late', media: false), isFalse);
      expect(pump.queued, 0);
    });
  });

  group('checkLoadedPass', () {
    Set<EmbeddingModality> check(
      EmbeddingPassInfo info,
      Set<EmbeddingModality> declared,
    ) => checkLoadedPass(info, declared, engine: 'Fake', label: 'm.bin');

    test('text tower + nothing declared = text', () {
      expect(check(const EmbeddingPassInfo(dimension: 4), const {}), {
        EmbeddingModality.text,
      });
    });

    test('declared media the engine serves are added to text', () {
      final info = EmbeddingPassInfo(
        dimension: 4,
        modalities: const {EmbeddingModality.text, EmbeddingModality.image},
        imageFormats: {EncodedMediaFormat.jpeg},
      );
      expect(check(info, const {EmbeddingModality.image}), {
        EmbeddingModality.text,
        EmbeddingModality.image,
      });
    });

    test('media the engine serves but nobody declared are NOT added', () {
      final info = EmbeddingPassInfo(
        dimension: 4,
        modalities: const {EmbeddingModality.text, EmbeddingModality.image},
        imageFormats: {EncodedMediaFormat.jpeg},
      );
      expect(check(info, const {}), {EmbeddingModality.text});
    });

    test('a declared modality the engine lacks is refused, named', () {
      expect(
        () => check(const EmbeddingPassInfo(dimension: 4), const {
          EmbeddingModality.image,
        }),
        throwsA(
          isA<EmbeddingLoadException>()
              .having((e) => e.engine, 'engine', 'Fake')
              .having((e) => e.message, 'message', contains('images'))
              .having((e) => e.message, 'message', contains('m.bin')),
        ),
      );
    });

    test('a non-positive dimension is refused', () {
      expect(
        () => check(const EmbeddingPassInfo(dimension: 0), const {}),
        throwsA(isA<EmbeddingLoadException>()),
      );
    });

    test('image support without image formats is an engine bug', () {
      expect(
        () => check(
          const EmbeddingPassInfo(
            dimension: 4,
            modalities: {EmbeddingModality.text, EmbeddingModality.image},
          ),
          const {},
        ),
        throwsA(isA<EmbeddingLoadException>()),
      );
    });

    test('audio support without audio formats is an engine bug', () {
      expect(
        () => check(
          const EmbeddingPassInfo(
            dimension: 4,
            modalities: {EmbeddingModality.audio},
          ),
          const {},
        ),
        throwsA(isA<EmbeddingLoadException>()),
      );
    });
  });

  group('EmbeddingFailure', () {
    EmbeddingException roundTrip(Object error, {bool loading = false}) =>
        EmbeddingFailure.fromError(
          error,
          engine: 'Fake',
          loading: loading,
        ).toException();

    test('closed', () {
      final e = roundTrip(const EmbedderClosedException('gone'));
      expect(e, isA<EmbedderClosedException>());
      expect(e.message, 'gone');
    });

    test('too long keeps both counts', () {
      expect(
        roundTrip(
          const EmbeddingInputTooLongException(
            'long',
            maxInputTokens: 8,
            actualTokens: 9,
          ),
        ),
        isA<EmbeddingInputTooLongException>()
            .having((e) => e.maxInputTokens, 'max', 8)
            .having((e) => e.actualTokens, 'actual', 9)
            .having((e) => e.message, 'message', 'long'),
      );
      expect(
        roundTrip(
          const EmbeddingInputTooLongException('long', maxInputTokens: 8),
        ),
        isA<EmbeddingInputTooLongException>().having(
          (e) => e.actualTokens,
          'actual',
          isNull,
        ),
      );
    });

    test('engine keeps its engine name and native code', () {
      expect(
        roundTrip(
          const EmbeddingEngineException('x', engine: 'Native', nativeCode: 3),
        ),
        isA<EmbeddingEngineException>()
            .having((e) => e.engine, 'engine', 'Native')
            .having((e) => e.nativeCode, 'code', 3),
      );
    });

    test('load keeps its engine name and native code', () {
      expect(
        roundTrip(
          const EmbeddingLoadException('x', engine: 'Native', nativeCode: 5),
        ),
        isA<EmbeddingLoadException>()
            .having((e) => e.engine, 'engine', 'Native')
            .having((e) => e.nativeCode, 'code', 5),
      );
    });

    test('anything else becomes an engine failure carrying its text', () {
      expect(
        roundTrip(StateError('kaput')),
        isA<EmbeddingEngineException>()
            .having((e) => e.engine, 'engine', 'Fake')
            .having((e) => e.message, 'message', 'Bad state: kaput'),
      );
    });

    test('anything else while loading becomes a load failure', () {
      expect(
        roundTrip(StateError('kaput'), loading: true),
        isA<EmbeddingLoadException>().having(
          (e) => e.message,
          'message',
          'Bad state: kaput',
        ),
      );
      expect(
        roundTrip(
          const EmbeddingEngineException('x', engine: 'Native'),
          loading: true,
        ),
        isA<EmbeddingLoadException>(),
      );
    });

    test('the cause is attached only where it is at hand', () {
      final failure = EmbeddingFailure.fromError(
        StateError('x'),
        engine: 'Fake',
      );
      expect(failure.toException().cause, isNull);
      expect(failure.toException(cause: 'native').cause, 'native');
    });
  });
}
