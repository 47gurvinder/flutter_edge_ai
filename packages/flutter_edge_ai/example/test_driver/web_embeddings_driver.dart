// Driver for the two web embedding drive suites
// (integration_test/web_embeddings_test.dart, web_embeddings_streaming_test.dart).
//
// Differs from test_driver/integration_test.dart in two ways: the response
// data is written EVEN WHEN A TEST FAILS — the W1 comparison and bundle API
// v1's ids are the record the MIGRATION re-index line is decided from, and a
// failing W1 is exactly when they are needed — and the run may take 45
// minutes, since the first run downloads EmbeddingGemma 1 (179 MB) and
// MiniLM (90 MB). Output: build/web_embeddings_response.json.
import 'package:integration_test/integration_test_driver.dart';

Future<void> main() => integrationDriver(
  timeout: const Duration(minutes: 45),
  writeResponseOnFailure: true,
  responseDataCallback: (data) =>
      writeResponseData(data, testOutputFilename: 'web_embeddings_response'),
);
