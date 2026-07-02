from evalkit.shell import VirtualShell


def sh(**kwargs):
    return VirtualShell(**kwargs)


def test_initial_state_and_pwd():
    s = sh(files={"a.txt": "hello\n", "sub/b.txt": "world\n"})
    assert s.run("pwd") == "/work"
    assert s.is_file("a.txt") and s.is_file("sub/b.txt")
    assert s.is_dir("sub")


def test_ls_and_cd():
    s = sh(files={"a.txt": "1", "sub/b.txt": "2"})
    assert s.run("ls") == "a.txt\nsub/"
    assert s.run("cd sub && pwd") == "/work/sub"
    assert s.run("ls") == "b.txt"
    assert s.run("cd .. && pwd") == "/work"
    assert "no such directory" in s.run("cd nope")


def test_cat_and_missing():
    s = sh(files={"a.txt": "hello\n"})
    assert s.run("cat a.txt") == "hello"
    assert "no such file" in s.run("cat nope.txt")


def test_echo_redirect_and_append():
    s = sh()
    s.run("echo 'line one' > f.txt")
    s.run("echo 'line two' >> f.txt")
    assert s.read("f.txt") == "line one\nline two\n"
    s.run("echo replaced > f.txt")
    assert s.read("f.txt") == "replaced\n"


def test_redirect_attached_form():
    s = sh()
    s.run("echo hi >f.txt")
    assert s.read("f.txt") == "hi\n"
    s.run("echo more >>f.txt")
    assert s.read("f.txt") == "hi\nmore\n"


def test_redirect_missing_parent_fails():
    s = sh()
    out = s.run("echo x > nodir/f.txt")
    assert "no such directory" in out
    assert not s.is_file("nodir/f.txt")


def test_mkdir_touch():
    s = sh()
    assert "use -p" in s.run("mkdir a/b")
    s.run("mkdir -p a/b/c")
    assert s.is_dir("a/b/c")
    s.run("touch a/b/c/f.txt")
    assert s.read("a/b/c/f.txt") == ""
    assert "no such directory" in s.run("touch zz/f.txt")


def test_rm_file_dir_force():
    s = sh(files={"f.txt": "x", "d/g.txt": "y"})
    assert "is a directory" in s.run("rm d")
    s.run("rm -r d")
    assert not s.is_dir("d") and not s.is_file("d/g.txt")
    assert "no such file" in s.run("rm nope")
    assert s.run("rm -f nope") == ""
    s.run("rm f.txt")
    assert not s.is_file("f.txt")


def test_rm_refuses_root():
    s = sh(files={"f.txt": "x"})
    assert "refusing" in s.run("rm -rf /")
    assert s.is_file("f.txt")


def test_mv_file_and_into_dir():
    s = sh(files={"a.txt": "content"}, dirs=["dest"])
    s.run("mv a.txt b.txt")
    assert s.read("b.txt") == "content" and not s.is_file("a.txt")
    s.run("mv b.txt dest")
    assert s.read("dest/b.txt") == "content"


def test_mv_directory():
    s = sh(files={"src/a.py": "code", "src/deep/b.py": "more"})
    s.run("mkdir -p lib && mv src lib")
    assert s.read("lib/src/a.py") == "code"
    assert s.read("lib/src/deep/b.py") == "more"
    assert not s.is_dir("src")


def test_cp_file_and_recursive():
    s = sh(files={"a.txt": "data", "d/x.txt": "1"})
    s.run("cp a.txt b.txt")
    assert s.read("a.txt") == "data" and s.read("b.txt") == "data"
    assert "is a directory" in s.run("cp d d2")
    s.run("cp -r d d2")
    assert s.read("d2/x.txt") == "1" and s.read("d/x.txt") == "1"


def test_grep_basics_and_exit_semantics():
    s = sh(files={"a.txt": "hello world\nfoo\n", "b.txt": "nothing\n"})
    assert s.run("grep foo a.txt") == "/work/a.txt:foo"
    assert s.run("grep -n hello a.txt") == "/work/a.txt:1:hello world"
    # no match -> failure -> '&&' chain short-circuits
    s.run("grep zebra a.txt && echo FOUND > marker.txt")
    assert not s.is_file("marker.txt")
    s.run("grep foo a.txt && echo FOUND > marker.txt")
    assert s.is_file("marker.txt")


def test_grep_recursive_and_invalid_regex_fallback():
    s = sh(files={"logs/a.log": "ok\n", "logs/b.log": "FATAL: dead\n"})
    assert "is a directory" in s.run("grep FATAL logs")
    assert s.run("grep -r FATAL logs") == "/work/logs/b.log:FATAL: dead"
    # '(' alone is an invalid regex; must fall back to literal matching
    s2 = sh(files={"f.txt": "call func( now\n"})
    assert "func(" in s2.run("grep 'func(' f.txt")


def test_head_and_wc():
    s = sh(files={"f.txt": "1\n2\n3\n4\n5\n"})
    assert s.run("head -n 2 f.txt") == "1\n2"
    assert s.run("wc -l f.txt") == "5 f.txt"


def test_chaining_semicolon_vs_and():
    s = sh()
    s.run("mkdir d ; echo hi > d/f.txt")
    assert s.read("d/f.txt") == "hi\n"
    # failed first command: ';' continues, '&&' stops
    s.run("cat nope.txt ; echo ok > a.txt")
    assert s.is_file("a.txt")
    s.run("cat nope.txt && echo ok > b.txt")
    assert not s.is_file("b.txt")


def test_quoted_arguments_protected():
    s = sh()
    s.run("echo 'a && b ; c' > f.txt")
    assert s.read("f.txt") == "a && b ; c\n"


def test_unknown_command_lists_supported():
    s = sh()
    out = s.run("sed -i s/a/b/ f.txt")
    assert "command not found" in out and "grep" in out


def test_output_truncation():
    s = sh(files={"big.txt": "x" * 10000})
    out = s.run("cat big.txt")
    assert len(out) < 10000 and "truncated" in out
