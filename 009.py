print("X1 | X2 | Yin | Y (ANDNOT)")
print("-" * 26)

for x1 in [0, 1]:
    for x2 in [0, 1]:
        yin = (x1 * 1) + (x2 * -1)
        y = 1 if yin >= 1 else 0
        print(f" {x1} |  {x2} | {yin:3} |     {y}")