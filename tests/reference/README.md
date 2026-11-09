# Reference tracks

The exactness gate needs local reference pairs:

```
tests/reference/<name>/
  audio.m4a
  ANLZ.DAT
  ANLZ.EXT   (optional; required for PQT2 microsecond comparison)
```

These files are deliberately not committed: they are user audio and
rekordbox-generated analysis data. Put them here locally to run:

```sh
python tests/verify_exact.py
```

The verifier uses the ANLZ only as expected output. It never passes the
reference grid to the generator.
