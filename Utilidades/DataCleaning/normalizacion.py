import unidecode

# Normalización de los nombres
def normalize_string(string: str) -> str:
    try:
        string = unidecode.unidecode(string.strip().lower())
        string = string.replace(".", "")
        string = string.replace(",", " ")
    except Exception as e:
        raise e

    return string

def normalize_personal_names(names):
    names = names[~names.isna()].apply(lambda name: normalize_string(name))

    to_remove = [
        "S.A.P.I.",
        "SA DE CV",
        "DE TOLUCA",
        "\*",
        "-",
    ]

    dict_to_remove = {normalize_string(item): "" for item in to_remove}
    dict_to_remove[normalize_string("C. PROPIETARIO")] = "propietario"
    dict_to_remove[normalize_string("C.PROPIETARIO")] = "propietario"
    dict_to_remove[r"\ss\s*a\s*d\s*e\s*c\s*v"] = " " # Regex para remover combinaciones de 'SA de CV'
    dict_to_remove[r"\ss\s*a($|\s)"] = " " # Remover combinaciones de 'SA'
    dict_to_remove['"'] = ""
    dict_to_remove[r"y\s*(\/|-)\s*(o|)"] = ""
    dict_to_remove["de cv"] = ""
    dict_to_remove["cv"] = ""
    dict_to_remove[r"\s+"] = " " # Regex para convertir espacios multiples a simples

    names = names.replace(dict_to_remove, regex=True)
    names = names[~names.isna()].apply(lambda name: normalize_string(name))

    return names

def normalize_names(names):
    names = names[~names.isna()].apply(lambda name: normalize_string(name))

    to_remove = [
        "'",
        "\[",
        " - ",
    ]

    dict_to_remove = {normalize_string(item): "" for item in to_remove}
    dict_to_remove[normalize_string("S/N")] = "SN"
    dict_to_remove[normalize_string("/C")] = "CALLE"
    dict_to_remove[normalize_string("\*")] = ""
    dict_to_remove[normalize_string("PRIVADA")] = "PRIV"
    dict_to_remove[normalize_string("PVADA.")] = "PRIV"
    dict_to_remove[normalize_string("PVADA")] = "PRIV"
    dict_to_remove[normalize_string("PVAD.")] = "PRIV"
    dict_to_remove[normalize_string("PRIV.")] = "PRIV"
    dict_to_remove[normalize_string("PVDA.")] = "PRIV"
    dict_to_remove[normalize_string("PVDA")] = "PRIV"
    dict_to_remove[normalize_string("PVA.")] = "PRIV"
    dict_to_remove[normalize_string("PVA")] = "PRIV"
    dict_to_remove[normalize_string("BOULEVARD.")] = "BLVR"
    dict_to_remove[normalize_string("BOULEVARD")] = "BLVR"
    dict_to_remove[normalize_string("BOULEVERD")] = "BLVR"
    dict_to_remove[normalize_string("BOULEBARD")] = "BLVR"
    dict_to_remove[normalize_string("BOUVEVARD")] = "BLVR"
    dict_to_remove[normalize_string("BAULEVAR")] = "BLVR"
    dict_to_remove[normalize_string("BOULEVAR")] = "BLVR"
    dict_to_remove[normalize_string("BOULEVAD")] = "BLVR"
    dict_to_remove[normalize_string("BOULVERD")] = "BLVR"
    dict_to_remove[normalize_string("BOULVARD")] = "BLVR"
    dict_to_remove[normalize_string("BOULBD.")] = "BLVR"
    dict_to_remove[normalize_string("BOULEV")] = "BLVR"
    dict_to_remove[normalize_string("BOULD")] = "BLVR"
    dict_to_remove[normalize_string("BOULBD")] = "BLVR"
    dict_to_remove[normalize_string("BOULVD")] = "BLVR"
    dict_to_remove[normalize_string("BOULV.")] = "BLVR"
    dict_to_remove[normalize_string("BOULV")] = "BLVR"

    dict_to_remove[normalize_string("1ERO")] = "1RO"
    dict_to_remove[normalize_string("1O.")] = "1RO"
    dict_to_remove[normalize_string("1O")] = "1RO"
    dict_to_remove[normalize_string("2NDA")] = "2A"
    dict_to_remove[normalize_string("2DA")] = "2A"
    
    dict_to_remove[normalize_string("AVENIDA")] = "AV"
    dict_to_remove[r"\s+"] = " " # Regex para convertir espacios multiples a simples

    names = names.replace(dict_to_remove, regex=True)
    names = names[~names.isna()].apply(lambda name: normalize_string(name))

    return names

def simplify_names(names):
    replace_table = str.maketrans(dict.fromkeys('aeiou. ')) # Definimos los caracteres que queremos quitar
    return names[~names.isna()].apply(lambda name: name.translate(replace_table))
 