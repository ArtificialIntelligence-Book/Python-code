# Given:
P_D = 0.01               # Probability disease present
P_Pos_given_D = 0.9     # Probability positive test given disease present (sensitivity)
P_Pos_given_notD = 0.05 # Probability positive test given disease not present (false positive rate)

# Total probability of positive test:
P_Pos = P_Pos_given_D * P_D + P_Pos_given_notD * (1 - P_D)

# Posterior probability using Bayes theorem:
P_D_given_Pos = (P_Pos_given_D * P_D) / P_Pos

print("Probability disease given positive test:", P_D_given_Pos)