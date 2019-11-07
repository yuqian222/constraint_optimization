\ Model mip1
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  slack0 + slack1 + slack2 + slack3 + slack4 + slack5 + slack6 + slack7
   + slack8 + slack9
Subject To
 R0: b0 + 0.19 C2 - 0.01 C3 - 0.34 C4 - b1 - 0.19 C7 + 0.01 C8 + 0.34 C9
   - slack0 >= 0
 R1: b0 - 0.16 C2 - 0.03 C3 + 0.23 C4 - b1 + 0.16 C7 + 0.03 C8 - 0.23 C9
   - slack1 >= 0
 R2: - b0 + 0.04 C1 + 0.16 C2 - 0.04 C3 - 0.34 C4 + b1 - 0.04 C6 - 0.16 C7
   + 0.04 C8 + 0.34 C9 - slack2 >= 0
 R3: - b0 - 0.03 C1 - 0.15 C2 - 0.02 C3 + 0.13 C4 + b1 + 0.03 C6 + 0.15 C7
   + 0.02 C8 - 0.13 C9 - slack3 >= 0
 R4: - b0 - 0.01 C1 - 0.19 C2 - 0.02 C3 + 0.15 C4 + b1 + 0.01 C6 + 0.19 C7
   + 0.02 C8 - 0.15 C9 - slack4 >= 0
 R5: - b0 - 0.03 C1 - 0.23 C2 - 0.04 C3 + 0.16 C4 + b1 + 0.03 C6 + 0.23 C7
   + 0.04 C8 - 0.16 C9 - slack5 >= 0
 R6: - b0 - 0.03 C1 - 0.03 C2 - 0.03 C3 - 0.1 C4 + b1 + 0.03 C6 + 0.03 C7
   + 0.03 C8 + 0.1 C9 - slack6 >= 0
 R7: b0 - 0.04 C2 + 0.03 C3 + 0.17 C4 - b1 + 0.04 C7 - 0.03 C8 - 0.17 C9
   - slack7 >= 0
 R8: - b0 + 0.04 C2 - 0.02 C3 - 0.11 C4 + b1 - 0.04 C7 + 0.02 C8 + 0.11 C9
   - slack8 >= 0
 R9: - b0 - 0.02 C1 + 0.15 C2 + 0.02 C3 - 0.24 C4 + b1 + 0.02 C6 - 0.15 C7
   - 0.02 C8 + 0.24 C9 - slack9 >= 0
Bounds
 -0.05 <= b0 <= 0.05
 -10 <= C1 <= 10
 -10 <= C2 <= 10
 -10 <= C3 <= 10
 -10 <= C4 <= 10
 -0.05 <= b1 <= 0.05
 -10 <= C6 <= 10
 -10 <= C7 <= 10
 -10 <= C8 <= 10
 -10 <= C9 <= 10
 0.0001 <= slack0 <= 0.1
 0.0001 <= slack1 <= 0.1
 0.0001 <= slack2 <= 0.1
 0.0001 <= slack3 <= 0.1
 0.0001 <= slack4 <= 0.1
 0.0001 <= slack5 <= 0.1
 0.0001 <= slack6 <= 0.1
 0.0001 <= slack7 <= 0.1
 0.0001 <= slack8 <= 0.1
 0.0001 <= slack9 <= 0.1
End
