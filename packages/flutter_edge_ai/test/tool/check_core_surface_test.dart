// Self-test of tool/check_core_surface.sh (R5 §2.3 item 3, §10.1 "Surface"):
// it runs the script over a fixture tree (CHECK_CORE_SURFACE_ROOT) holding
// one violation of each rule, and over a clean tree. A guard that has never
// been seen failing is not known to work.
@TestOn('vm')
library;

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

/// From this package's directory, where `tool/test_all.sh` runs the suite.
final _script = File('../../tool/check_core_surface.sh').absolute.path;

void _write(Directory root, String relative, String content) {
  File('${root.path}/$relative')
    ..createSync(recursive: true)
    ..writeAsStringSync(content);
}

/// A minimal workspace: core with one private file, and a satellite.
Directory _cleanTree() {
  final root = Directory.systemTemp.createTempSync('core_surface_');
  _write(
    root,
    'packages/flutter_edge_ai/pubspec.yaml',
    'name: flutter_edge_ai\n',
  );
  _write(
    root,
    'packages/flutter_edge_ai/lib/flutter_edge_ai.dart',
    'library;\n',
  );
  _write(root, 'packages/flutter_edge_ai/lib/src/x.dart', 'const x = 1;\n');
  _write(
    root,
    'packages/flutter_edge_ai/example/pubspec.yaml',
    'name: flutter_edge_ai_example\n',
  );
  _write(
    root,
    'packages/flutter_edge_ai/example/lib/main.dart',
    "import 'package:flutter_edge_ai/flutter_edge_ai.dart';\n",
  );
  _write(root, 'packages/sat/pubspec.yaml', 'name: sat\n');
  _write(
    root,
    'packages/sat/lib/sat.dart',
    "import 'package:flutter_edge_ai/engine.dart';\n",
  );
  _write(
    root,
    'packages/sat/test/white_box_test.dart',
    '// ignore: implementation_imports — reads a private constant\n'
        "import 'package:flutter_edge_ai/src/x.dart';\n",
  );
  return root;
}

Future<ProcessResult> _run(Directory root) => Process.run(
  'bash',
  [_script],
  environment: {'CHECK_CORE_SURFACE_ROOT': root.path},
);

void main() {
  test('a clean tree passes and lists the annotated import', () async {
    final root = _cleanTree();
    addTearDown(() => root.deleteSync(recursive: true));

    final result = await _run(root);
    expect(result.exitCode, 0, reason: '${result.stdout}${result.stderr}');
    expect(result.stdout, contains('white_box_test.dart'));
    expect(result.stdout, contains('reads a private constant'));
  });

  test('each rule fails on its own violation', () async {
    final root = _cleanTree();
    addTearDown(() => root.deleteSync(recursive: true));
    // (a) a deep path that is neither public nor under src/.
    _write(
      root,
      'packages/sat/lib/a.dart',
      "import 'package:flutter_edge_ai/core/old_path.dart';\n",
    );
    // (b) a src/ import in a test without the annotation.
    _write(
      root,
      'packages/sat/test/b_test.dart',
      "import 'package:flutter_edge_ai/src/x.dart';\n",
    );
    // (c) a src/ import in shipped code, annotated or not.
    _write(
      root,
      'packages/sat/lib/c.dart',
      '// ignore: implementation_imports — no exception for lib/\n'
          "import 'package:flutter_edge_ai/src/x.dart';\n",
    );
    // (c) also an example app's lib/, which ships in the pub archive — the
    // carve-out that once excused one example file is gone.
    _write(
      root,
      'packages/flutter_edge_ai/example/lib/c_example.dart',
      '// ignore: implementation_imports — no exception for example/lib\n'
          "import 'package:flutter_edge_ai/src/x.dart';\n",
    );

    final result = await _run(root);
    final out = '${result.stdout}';
    expect(result.exitCode, 1, reason: out);
    expect(out, contains(RegExp(r'FAIL\s+packages/sat/lib/a\.dart:1\s+\(a\)')));
    expect(
      out,
      contains(RegExp(r'FAIL\s+packages/sat/test/b_test\.dart:1\s+\(b\)')),
    );
    expect(out, contains(RegExp(r'FAIL\s+packages/sat/lib/c\.dart:2\s+\(c\)')));
    expect(
      out,
      contains(
        RegExp(
          r'FAIL\s+packages/flutter_edge_ai/example/lib/c_example\.dart:2\s+\(c\)',
        ),
      ),
    );
    expect(out, contains('4 violation(s)'));
  });

  test('a tree with nothing to scan is not a pass', () async {
    final root = Directory.systemTemp.createTempSync('core_surface_empty_');
    addTearDown(() => root.deleteSync(recursive: true));
    Directory('${root.path}/packages').createSync();

    final result = await _run(root);
    expect(result.exitCode, 1);
  });
}
