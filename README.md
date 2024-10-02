# Field Configuration Utilities

Field configurations are specified as YAML files with the following format:

```yaml
rows:
  row1:
  row2:
    
  rowN:
```

The field is specified as a dictionary of rows. Each individual row can have 
any name you want, but they should be specified from west to east (if the 
field has a vertical layout), or from north to south (if it has a horizontal 
one.)

The most basic way to specify a row is just to give it a list of plot 
numbers in that row:

```yaml
row1:
  - 1001
  - 1002
  - 1003
  - 1004
  - 1005
```

By convention, plot numbers are specified from north to south (for vertical 
fields), or west to east (for horizontal fields).

Since plots are often sequentially numbered, the `range` attribute can be 
used to specify this compactly. Using `range`, the above is equivalent to:

```yaml
row1:
  - range:
    start: 1001
    end: 1005
```

`range` also supports a `repeats` attribute that specifies how many times 
you want to repeat each plot number.

Additionally, you can construct a new row based on a previous row with all 
the plot numbers shifted by some constant. For example, the following:

```yaml
row2:
  - shift:
    row: row1
    amount: 5
```

is equivalent to

```yaml
row2:
  - range:
    start: 1006
    end: 1010
```

All of these features can be combined in one row to specify the plots in the 
row in order. For example:

```yaml
row3:
  - 100
  - range:
    start: 1201
    end: 1205
  - 101
```

is equivalent to

```yaml
row3:
  - 100
  - 1201
  - 1202
  - 1203
  - 1204
  - 1205
  - 101
```
