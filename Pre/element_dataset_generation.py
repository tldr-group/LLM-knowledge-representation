import pandas as pd
import os

data = {
    'Element Name': ['Hydrogen', 'Helium', 'Lithium', 'Beryllium', 'Boron', 'Carbon', 'Nitrogen', 'Oxygen', 'Fluorine', 'Neon',
                     'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Phosphorus', 'Sulfur', 'Chlorine', 'Argon', 'Potassium', 'Calcium',
                     'Scandium', 'Titanium', 'Vanadium', 'Chromium', 'Manganese', 'Iron', 'Cobalt', 'Nickel', 'Copper', 'Zinc', 
                     'Gallium', 'Germanium', 'Arsenic', 'Selenium', 'Bromine', 'Krypton', 'Rubidium', 'Strontium', 'Yttrium', 'Zirconium', 
                     'Niobium', 'Molybdenum', 'Technetium', 'Ruthenium', 'Rhodium', 'Palladium', 'Silver', 'Cadmium', 'Indium', 'Tin'],
    'Atomic Number': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                      31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                      41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
    'Symbol': ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
               'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
               'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
               'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
               'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn'],
    'Atomic Mass': [1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180,
                    22.990, 24.305, 26.982, 28.085, 30.974, 32.06, 35.45, 39.948, 39.098, 40.078,
                    44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.38,
                    69.723, 72.63, 74.922, 78.971, 79.904, 83.798, 85.468, 87.62, 88.906, 91.224,
                    92.906, 95.95, 98, 101.07, 102.91, 106.42, 107.87, 112.41, 114.82, 118.71],
    'Electron Configuration': ['1s1', '1s2', '[He] 2s1', '[He] 2s2', '[He] 2s2 2p1', '[He] 2s2 2p2', '[He] 2s2 2p3', '[He] 2s2 2p4', '[He] 2s2 2p5', '[He] 2s2 2p6',
                               '[Ne] 3s1', '[Ne] 3s2', '[Ne] 3s2 3p1', '[Ne] 3s2 3p2', '[Ne] 3s2 3p3', '[Ne] 3s2 3p4', '[Ne] 3s2 3p5', '[Ne] 3s2 3p6', '[Ar] 4s1', '[Ar] 4s2',
                               '[Ar] 3d1 4s2', '[Ar] 3d2 4s2', '[Ar] 3d3 4s2', '[Ar] 3d5 4s1', '[Ar] 3d5 4s2', '[Ar] 3d6 4s2', '[Ar] 3d7 4s2', '[Ar] 3d8 4s2', '[Ar] 3d10 4s1', '[Ar] 3d10 4s2',
                               '[Ar] 3d10 4s2 4p1', '[Ar] 3d10 4s2 4p2', '[Ar] 3d10 4s2 4p3', '[Ar] 3d10 4s2 4p4', '[Ar] 3d10 4s2 4p5', '[Ar] 3d10 4s2 4p6', '[Kr] 5s1', '[Kr] 5s2', '[Kr] 4d1 5s2', '[Kr] 4d2 5s2',
                               '[Kr] 4d4 5s1', '[Kr] 4d5 5s1', '[Kr] 4d5 5s2', '[Kr] 4d7 5s1', '[Kr] 4d8 5s1', '[Kr] 4d10', '[Kr] 4d10 5s1', '[Kr] 4d10 5s2', '[Kr] 4d10 5s2 5p1', '[Kr] 4d10 5s2 5p2'],
    'Electronegativity': [2.20, None, 0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, None,
                          0.93, 1.31, 1.61, 1.90, 2.19, 2.58, 3.16, None, 0.82, 1.00,
                          1.36, 1.54, 1.63, 1.66, 1.55, 1.83, 1.88, 1.91, 1.90, 1.65,
                          1.81, 2.01, 2.18, 2.55, 2.96, None, 0.82, 0.95, 1.22, 1.33,
                          1.6, 2.16, None, 2.20, 2.28, 2.20, 1.93, 1.69, 1.78, 1.96],
    'Oxidation States': [1, 0, 1, 2, 3, 4, 3, -2, -1, 0,
                         1, 2, 3, 4, 3, 2, -1, 0, 1, 2,
                         3, 4, 5, 2, 7, 2, 3, 2, 2, 2,
                         3, 4, 5, -2, -1, 0, 1, 2, 3, 4,
                         5, 6, 7, 8, 3, 2, 1, 2, 3, 4],
    'Group': [1, 18, 1, 2, 13, 14, 15, 16, 17, 18,
              1, 2, 13, 14, 15, 16, 17, 18, 1, 2,
              3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
              13, 14, 15, 16, 17, 18, 1, 2, 3, 4,
              5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    'Period': [1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
               3, 3, 3, 3, 3, 3, 3, 3, 4, 4,
               4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
               4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
               5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    'Is Metal': [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 
                    1, 1, 1, 0, 0, 0, 0, 0, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      "Next Period": [3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                  19, 20, 31, 32, 33, 34, 35, 36, 37, 38, 
                  39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 
                  49, 50, None, None, None, None, None, None, 
                  None, None, None, None, None, None, None, None, 
                  None, None, None, None],

    'Category_2': [
    'Non-metal',           # 1. Hydrogen
    'Noble gas',           # 2. Helium
    'Metal',               # 3. Lithium
    'Metal',               # 4. Beryllium
    'Non-metal',           # 5. Boron
    'Non-metal',           # 6. Carbon
    'Non-metal',           # 7. Nitrogen
    'Non-metal',           # 8. Oxygen
    'Halogen',             # 9. Fluorine
    'Noble gas',           # 10. Neon
    'Metal',               # 11. Sodium
    'Metal',               # 12. Magnesium
    'Metal',               # 13. Aluminum
    'Non-metal',           # 14. Silicon
    'Non-metal',           # 15. Phosphorus
    'Non-metal',           # 16. Sulfur
    'Halogen',             # 17. Chlorine
    'Noble gas',           # 18. Argon
    'Metal',               # 19. Potassium
    'Metal',               # 20. Calcium
    'Metal',               # 21. Scandium
    'Metal',               # 22. Titanium
    'Metal',               # 23. Vanadium
    'Metal',               # 24. Chromium
    'Metal',               # 25. Manganese
    'Metal',               # 26. Iron
    'Metal',               # 27. Cobalt
    'Metal',               # 28. Nickel
    'Metal',               # 29. Copper
    'Metal',               # 30. Zinc
    'Metal',               # 31. Gallium
    'Metal',               # 32. Germanium
    'Non-metal',           # 33. Arsenic
    'Non-metal',           # 34. Selenium
    'Halogen',             # 35. Bromine
    'Noble gas',           # 36. Krypton
    'Metal',               # 37. Rubidium
    'Metal',               # 38. Strontium
    'Metal',               # 39. Yttrium
    'Metal',               # 40. Zirconium
    'Metal',               # 41. Niobium
    'Metal',               # 42. Molybdenum
    'Metal',               # 43. Technetium
    'Metal',               # 44. Ruthenium
    'Metal',               # 45. Rhodium
    'Metal',               # 46. Palladium
    'Metal',               # 47. Silver
    'Metal',               # 48. Cadmium
    'Metal',               # 49. Indium
    'Metal'                # 50. Tin
],

'Category': [
    'Non-metal',           # 1. Hydrogen
    'Noble gas',           # 2. Helium
    'Alkali metal',        # 3. Lithium
    'Alkaline earth metal',# 4. Beryllium
    'Metalloid',           # 5. Boron
    'Non-metal',           # 6. Carbon
    'Non-metal',           # 7. Nitrogen
    'Non-metal',           # 8. Oxygen
    'Halogen',             # 9. Fluorine
    'Noble gas',           # 10. Neon
    'Alkali metal',        # 11. Sodium
    'Alkaline earth metal',# 12. Magnesium
    'Post-transition metal',# 13. Aluminum
    'Metalloid',           # 14. Silicon
    'Non-metal',           # 15. Phosphorus
    'Non-metal',           # 16. Sulfur
    'Halogen',             # 17. Chlorine
    'Noble gas',           # 18. Argon
    'Alkali metal',        # 19. Potassium
    'Alkaline earth metal',# 20. Calcium
    'Transition metal',    # 21. Scandium
    'Transition metal',    # 22. Titanium
    'Transition metal',    # 23. Vanadium
    'Transition metal',    # 24. Chromium
    'Transition metal',    # 25. Manganese
    'Transition metal',    # 26. Iron
    'Transition metal',    # 27. Cobalt
    'Transition metal',    # 28. Nickel
    'Transition metal',    # 29. Copper
    'Transition metal',    # 30. Zinc
    'Post-transition metal',# 31. Gallium
    'Metalloid',           # 32. Germanium
    'Metalloid',           # 33. Arsenic
    'Non-metal',           # 34. Selenium
    'Halogen',             # 35. Bromine
    'Noble gas',           # 36. Krypton
    'Alkali metal',        # 37. Rubidium
    'Alkaline earth metal',# 38. Strontium
    'Transition metal',    # 39. Yttrium
    'Transition metal',    # 40. Zirconium
    'Transition metal',    # 41. Niobium
    'Transition metal',    # 42. Molybdenum
    'Transition metal',    # 43. Technetium
    'Transition metal',    # 44. Ruthenium
    'Transition metal',    # 45. Rhodium
    'Transition metal',    # 46. Palladium
    'Transition metal',    # 47. Silver
    'Post-transition metal',# 48. Cadmium
    'Post-transition metal',# 49. Indium
    'Post-transition metal' # 50. Tin

]


}

lengths = {key: len(value) for key, value in data.items()}
print(lengths)
elements_df = pd.DataFrame(data)

elements_df.to_csv('periodic_table_dataset.csv', index=False)
