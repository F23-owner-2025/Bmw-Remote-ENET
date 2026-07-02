"""VirtualShell — a deterministic, in-memory POSIX-ish shell for agentic evals.

Why simulated instead of real bash in a tempdir: grading needs to be exactly
reproducible, safe against anything the model emits (`rm -rf /`, forkbombs),
and cheap enough to run thousands of episodes — the same properties an RL
environment needs, which this class becomes in a v2. The command surface is
deliberately small, documented to the model in its system prompt, and every
command is unit-tested.

Supported commands:
    pwd, cd, ls, cat, echo (-n), touch, mkdir (-p), rm (-r, -f), mv,
    cp (-r), grep (-r, -n), head (-n K), wc (-l)

Semantics implemented: `&&` / `;` chaining (quote-aware via shlex), `>` and
`>>` redirection of echo/cat/grep/head output. grep exits nonzero on no
match, like the real one — so `grep X f && rm f` behaves correctly.
"""

from __future__ import annotations

import posixpath
import re
import shlex
from typing import Dict, Iterable, List, Optional, Tuple

MAX_RUN_OUTPUT = 4000

COMMAND_DOC = (
    "pwd, cd, ls, cat, echo, touch, mkdir -p, rm [-r] [-f], mv, cp [-r], "
    "grep [-r] [-n], head [-n K], wc -l. Chaining with '&&' and ';' and "
    "redirection with '>' / '>>' are supported. There is no pipe support."
)


class VirtualShell:
    def __init__(self, files: Optional[Dict[str, str]] = None,
                 dirs: Optional[Iterable[str]] = None, cwd: str = "/work"):
        self.files: Dict[str, str] = {}
        self.dirs = {"/"}
        self.cwd = "/"
        self._mkdir_p(self._resolve(cwd))
        self.cwd = self._resolve(cwd)
        for d in dirs or []:
            self._mkdir_p(self._resolve(d))
        for path, content in (files or {}).items():
            ap = self._resolve(path)
            self._mkdir_p(posixpath.dirname(ap))
            self.files[ap] = content

    # ------------------------------------------------------------------ paths

    def _resolve(self, path: str) -> str:
        if not path.startswith("/"):
            path = posixpath.join(self.cwd, path)
        return posixpath.normpath(path)

    def _mkdir_p(self, ap: str) -> None:
        self.dirs.add("/")
        parts = [p for p in ap.split("/") if p]
        cur = ""
        for p in parts:
            cur += "/" + p
            self.dirs.add(cur)

    def is_dir(self, path: str) -> bool:
        return self._resolve(path) in self.dirs

    def is_file(self, path: str) -> bool:
        return self._resolve(path) in self.files

    def read(self, path: str) -> Optional[str]:
        return self.files.get(self._resolve(path))

    def _children(self, ap: str) -> List[str]:
        prefix = "/" if ap == "/" else ap + "/"
        names = set()
        for f in self.files:
            if f.startswith(prefix):
                rel = f[len(prefix):]
                if "/" not in rel:  # direct child files only; dirs come below
                    names.add(rel)
        for d in self.dirs:
            if d != ap and d.startswith(prefix):
                head = d[len(prefix):].split("/")[0]
                names.add(head + "/")
        return sorted(names)

    # ------------------------------------------------------------------ run

    def run(self, script: str) -> str:
        """Execute a script (possibly multiple lines/chains); return output."""
        outputs: List[str] = []
        for line in script.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                tokens = shlex.split(line, posix=True)
            except ValueError as err:
                outputs.append(f"sh: parse error: {err}")
                continue
            for ok, out in self._run_chains(tokens):
                if out:
                    outputs.append(out)
        result = "\n".join(outputs)
        if len(result) > MAX_RUN_OUTPUT:
            result = result[:MAX_RUN_OUTPUT] + "\n... [output truncated]"
        return result

    def _run_chains(self, tokens: List[str]) -> List[Tuple[bool, str]]:
        # Split token stream on '&&' and ';', remembering each separator.
        segments: List[Tuple[str, List[str]]] = []  # (sep_before, tokens)
        cur: List[str] = []
        sep = ";"
        for tok in tokens:
            if tok in ("&&", ";"):
                if cur:
                    segments.append((sep, cur))
                sep, cur = tok, []
            else:
                cur.append(tok)
        if cur:
            segments.append((sep, cur))

        results: List[Tuple[bool, str]] = []
        prev_ok = True
        for sep_before, seg in segments:
            if sep_before == "&&" and not prev_ok:
                continue  # short-circuit the chain
            ok, out = self._run_one(seg)
            results.append((ok, out))
            prev_ok = ok
        return results

    def _run_one(self, tokens: List[str]) -> Tuple[bool, str]:
        tokens, redirect = self._extract_redirect(tokens)
        if not tokens:
            return False, "sh: empty command"
        name, args = tokens[0], tokens[1:]
        handler = getattr(self, f"_cmd_{name}", None)
        if handler is None:
            return False, (f"sh: {name}: command not found "
                           f"(supported: {COMMAND_DOC})")
        ok, out = handler(args)
        if ok and redirect is not None:
            mode, target = redirect
            ap = self._resolve(target)
            parent = posixpath.dirname(ap)
            if parent not in self.dirs:
                return False, f"sh: {target}: no such directory"
            if ap in self.dirs:
                return False, f"sh: {target}: is a directory"
            text = out + ("\n" if out and not out.endswith("\n") else "")
            if mode == ">>" and ap in self.files:
                self.files[ap] += text
            else:
                self.files[ap] = text
            return True, ""
        return ok, out

    @staticmethod
    def _extract_redirect(tokens: List[str]):
        out_tokens: List[str] = []
        redirect: Optional[Tuple[str, str]] = None
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in (">", ">>"):
                if i + 1 >= len(tokens):
                    return out_tokens, None
                redirect = (tok, tokens[i + 1])
                i += 2
            elif tok.startswith(">>"):
                redirect = (">>", tok[2:])
                i += 1
            elif tok.startswith(">") and len(tok) > 1:
                redirect = (">", tok[1:])
                i += 1
            else:
                out_tokens.append(tok)
                i += 1
        return out_tokens, redirect

    # ------------------------------------------------------------------ cmds

    def _cmd_pwd(self, args):
        return True, self.cwd

    def _cmd_cd(self, args):
        target = args[0] if args else "/"
        ap = self._resolve(target)
        if ap not in self.dirs:
            return False, f"sh: cd: {target}: no such directory"
        self.cwd = ap
        return True, ""

    def _cmd_ls(self, args):
        paths = [a for a in args if not a.startswith("-")] or ["."]
        chunks = []
        ok = True
        for p in paths:
            ap = self._resolve(p)
            if ap in self.dirs:
                chunks.append("\n".join(self._children(ap)))
            elif ap in self.files:
                chunks.append(p)
            else:
                ok = False
                chunks.append(f"sh: ls: {p}: no such file or directory")
        return ok, "\n".join(c for c in chunks if c)

    def _cmd_cat(self, args):
        parts = []
        for p in args:
            if not p.startswith("-"):
                ap = self._resolve(p)
                if ap not in self.files:
                    return False, f"sh: cat: {p}: no such file"
                parts.append(self.files[ap])
        return True, "".join(parts).rstrip("\n")

    def _cmd_echo(self, args):
        if args and args[0] == "-n":
            args = args[1:]
        return True, " ".join(args)

    def _cmd_touch(self, args):
        for p in args:
            ap = self._resolve(p)
            parent = posixpath.dirname(ap)
            if parent not in self.dirs:
                return False, f"sh: touch: {p}: no such directory"
            self.files.setdefault(ap, "")
        return True, ""

    def _cmd_mkdir(self, args):
        recursive = "-p" in args
        for p in (a for a in args if not a.startswith("-")):
            ap = self._resolve(p)
            parent = posixpath.dirname(ap)
            if not recursive and parent not in self.dirs:
                return False, f"sh: mkdir: {p}: no such directory (use -p)"
            if ap in self.files:
                return False, f"sh: mkdir: {p}: file exists"
            self._mkdir_p(ap)
        return True, ""

    def _cmd_rm(self, args):
        flags = {a for a in args if a.startswith("-")}
        recursive = bool(flags & {"-r", "-rf", "-fr", "-R"})
        force = bool(flags & {"-f", "-rf", "-fr"})
        for p in (a for a in args if not a.startswith("-")):
            ap = self._resolve(p)
            if ap in self.files:
                del self.files[ap]
            elif ap in self.dirs:
                if not recursive:
                    return False, f"sh: rm: {p}: is a directory (use -r)"
                if ap == "/":
                    return False, "sh: rm: refusing to remove /"
                prefix = ap + "/"
                self.files = {f: c for f, c in self.files.items()
                              if not f.startswith(prefix)}
                self.dirs = {d for d in self.dirs
                             if d != ap and not d.startswith(prefix)}
                if self.cwd == ap or self.cwd.startswith(prefix):
                    self.cwd = "/"
            elif not force:
                return False, f"sh: rm: {p}: no such file or directory"
        return True, ""

    def _move_or_copy(self, args, verb: str, keep_source: bool):
        flags = {a for a in args if a.startswith("-")}
        recursive = bool(flags & {"-r", "-R"}) or verb == "mv"
        paths = [a for a in args if not a.startswith("-")]
        if len(paths) != 2:
            return False, f"sh: {verb}: expected SOURCE DEST"
        src, dst = (self._resolve(p) for p in paths)
        if src in self.files:
            dst_final = (posixpath.join(dst, posixpath.basename(src))
                         if dst in self.dirs else dst)
            parent = posixpath.dirname(dst_final)
            if parent not in self.dirs:
                return False, f"sh: {verb}: {paths[1]}: no such directory"
            self.files[dst_final] = self.files[src]
            if not keep_source:
                del self.files[src]
            return True, ""
        if src in self.dirs:
            if not recursive:
                return False, f"sh: {verb}: {paths[0]}: is a directory (use -r)"
            dst_final = (posixpath.join(dst, posixpath.basename(src))
                         if dst in self.dirs else dst)
            parent = posixpath.dirname(dst_final)
            if parent not in self.dirs:
                return False, f"sh: {verb}: {paths[1]}: no such directory"
            prefix = src + "/"
            moves = [(f, dst_final + "/" + f[len(prefix):])
                     for f in list(self.files) if f.startswith(prefix)]
            sub_dirs = [d for d in list(self.dirs)
                        if d == src or d.startswith(prefix)]
            self._mkdir_p(dst_final)
            for d in sub_dirs:
                rel = d[len(src):]
                self._mkdir_p(dst_final + rel)
            for old, new in moves:
                self.files[new] = self.files[old]
                if not keep_source:
                    del self.files[old]
            if not keep_source:
                self.dirs = {d for d in self.dirs
                             if d != src and not d.startswith(prefix)}
            return True, ""
        return False, f"sh: {verb}: {paths[0]}: no such file or directory"

    def _cmd_mv(self, args):
        return self._move_or_copy(args, "mv", keep_source=False)

    def _cmd_cp(self, args):
        return self._move_or_copy(args, "cp", keep_source=True)

    def _cmd_grep(self, args):
        flags = {a for a in args if a.startswith("-") and a != "-"}
        recursive = bool(flags & {"-r", "-R", "-rn", "-nr"})
        show_lineno = bool(flags & {"-n", "-rn", "-nr"})
        rest = [a for a in args if a not in flags]
        if not rest:
            return False, "sh: grep: usage: grep [-r] [-n] PATTERN PATH..."
        pattern, paths = rest[0], rest[1:] or ["."]
        try:
            rx = re.compile(pattern)
            match = rx.search
        except re.error:
            match = lambda line: pattern in line  # noqa: E731

        targets: List[str] = []
        for p in paths:
            ap = self._resolve(p)
            if ap in self.files:
                targets.append(ap)
            elif ap in self.dirs:
                if not recursive:
                    return False, f"sh: grep: {p}: is a directory (use -r)"
                prefix = "/" if ap == "/" else ap + "/"
                targets.extend(sorted(f for f in self.files
                                      if f.startswith(prefix)))
            else:
                return False, f"sh: grep: {p}: no such file or directory"

        lines_out = []
        for f in targets:
            for i, line in enumerate(self.files[f].splitlines(), start=1):
                if match(line):
                    loc = f"{f}:{i}:" if show_lineno else f"{f}:"
                    lines_out.append(f"{loc}{line}")
        # Real grep exits 1 on no match — keeps '&&' chains honest.
        return (True, "\n".join(lines_out)) if lines_out else (False, "")

    def _cmd_head(self, args):
        n = 10
        rest = []
        i = 0
        while i < len(args):
            if args[i] == "-n" and i + 1 < len(args):
                try:
                    n = int(args[i + 1])
                except ValueError:
                    return False, f"sh: head: bad line count {args[i + 1]}"
                i += 2
            else:
                rest.append(args[i])
                i += 1
        parts = []
        for p in rest:
            ap = self._resolve(p)
            if ap not in self.files:
                return False, f"sh: head: {p}: no such file"
            parts.append("\n".join(self.files[ap].splitlines()[:n]))
        return True, "\n".join(parts)

    def _cmd_wc(self, args):
        rest = [a for a in args if not a.startswith("-")]
        if "-l" not in args:
            return False, "sh: wc: only 'wc -l FILE' is supported"
        out = []
        for p in rest:
            ap = self._resolve(p)
            if ap not in self.files:
                return False, f"sh: wc: {p}: no such file"
            count = len(self.files[ap].splitlines())
            out.append(f"{count} {p}")
        return True, "\n".join(out)
