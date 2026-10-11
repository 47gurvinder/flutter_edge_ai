// The inline host's per-request yield on the web page: a MessageChannel
// message. Like a timer it is a task, so whatever is already queued — the
// timer or input event a `close()` is called from — runs before it, and the
// pump sees that close before it starts the next request. Unlike a timer it is
// not clamped: browsers hold nested zero-delay timers to >= 4 ms, and with a
// zero timer here `embedAll` of 200 short texts over an instant pass took
// 864–934 ms in Chrome 155 (≈ 4.3 ms per request, R5_adjustments 2026-10-11).
//
// dart:js_interop only, so the web graph stays free of dart:io/ffi/isolate.

import 'dart:async';
import 'dart:collection';
import 'dart:js_interop';

@JS('MessageChannel')
extension type _MessageChannel._(JSObject _) implements JSObject {
  external factory _MessageChannel();
  external _MessagePort get port1;
  external _MessagePort get port2;
}

extension type _MessagePort._(JSObject _) implements JSObject {
  external set onmessage(JSFunction? handler);
  external void postMessage(JSAny? message);
}

/// One channel for the page: messages on a port arrive in the order they were
/// posted, so each one completes the oldest waiter.
final _waiting = Queue<Completer<void>>();
_MessagePort? _sender;

_MessagePort _channel() {
  final existing = _sender;
  if (existing != null) return existing;
  final channel = _MessageChannel();
  channel.port1.onmessage = ((JSAny? _) {
    _waiting.removeFirst().complete();
  }).toJS;
  return _sender = channel.port2;
}

/// Completes after the tasks queued before the call have run.
Future<void> yieldToEventLoop() {
  final turn = Completer<void>();
  _waiting.add(turn);
  _channel().postMessage(null);
  return turn.future;
}
