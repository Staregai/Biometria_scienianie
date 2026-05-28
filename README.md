# Biometria P3 - porownanie algorytmow scieniania odciskow palcow

Projekt realizuje wymagania z `BIO_2026_Projekt_3.pdf`:

- porownanie dwoch algorytmow scieniania,
- szkieletyzacja morfologiczna,
- scienianie `K3M` i `KMM`,
- dodatkowe operacje poprawiajace ciaglosc linii papilarnych,
- lokalizacja minucji (`zakonczenia`, `bifurkacje`),
- adaptacyjna binarizacja (Sauvola) i maskowanie obszaru odcisku,
- desktopowe GUI w `tkinter` do ogladania wszystkich etapow i eksportu wynikow.

## Pliki

- `app.py` - aplikacja desktopowa,
- `fingerprint_processor.py` - implementacja przetwarzania i analizy,
- `requirements.txt` - zaleznosci runtime.

## Uruchomienie

```bash
pip install -r requirements.txt
python app.py
```

## Dane

Przykladowe skany sa w folderze:

`skaner_odciskow/en/Demo`

## Pipeline

1. Wczytanie i wzmocnienie kontrastu odcisku.
2. Binaryzacja Otsu lub Sauvola (adaptacyjna), opcjonalnie invert.
3. Oczyszczenie obrazu morfologia i maska obszaru odcisku.
4. Lączenie krotkich przerw miedzy grzbietami oraz laczenie endpointow.
5. Scienianie:
   - morfologiczne,
   - `K3M`,
   - `KMM`.
6. Redukcja drobnych odnog (dlugosc) i detekcja minucji metoda `crossing number`.

## Uwagi

- Implementacja korzysta tylko z `Pillow` do IO i rysowania podgladow.
- Algorytmy przetwarzania i scieniania sa zaimplementowane recznie w Pythonie.
