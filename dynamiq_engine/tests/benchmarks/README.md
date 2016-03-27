This directory includes several scripts for benchmarking/profiling the
dynamiq engine. 

```sh
python profiling.py
gprof2dot -f pstats morse.pstats| dot -Tpng -o morse.png
```
