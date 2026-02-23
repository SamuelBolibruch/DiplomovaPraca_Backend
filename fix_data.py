import csv
import os

def fix_biometry_csv(input_path):
    # Zmeníme output_path tak, aby bola rovnaká ako input_path
    output_path = input_path 
    fixed_rows = []
    
    if not os.path.exists(input_path):
        print(f"Chyba: Súbor {input_path} neexistuje!")
        return

    # 1. Čítanie dát do pamäte
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            fixed_rows.append(header)
        except StopIteration:
            return 

        for row in reader:
            if len(row) > 9:
                start = row[:6]
                end = row[-2:]
                middle_val = ",".join(row[6:-2])
                if middle_val == "":
                    middle_val = "," * (len(row) - 8)
                
                new_row = start + [middle_val] + end
                fixed_rows.append(new_row)
            else:
                fixed_rows.append(row)

    # 2. Zápis späť do TOHO ISTÉHO súboru
    # Režim 'w' vymaže starý obsah a zapíše nový (opravený)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(fixed_rows)
    
    print(f"✅ Hotovo! Pôvodný súbor bol aktualizovaný: {output_path}")
    return output_path

# --- TU ZADAJ SVOJU CESTU ---
moja_cesta = 'data/samuelbolibruch/keystrokes_common.csv'
fix_biometry_csv(moja_cesta)