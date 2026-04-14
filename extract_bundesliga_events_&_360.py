import argparse
import json
from pathlib import Path
from typing import Set


def cargar_match_ids(matches_file: Path) -> Set[int]:
    with matches_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("El archivo de matches debe contener una lista JSON.")

    match_ids: Set[int] = set()
    for item in data:
        if isinstance(item, dict) and "match_id" in item:
            try:
                match_ids.add(int(item["match_id"]))
            except (TypeError, ValueError):
                continue

    if not match_ids:
        raise ValueError("No se encontraron match_id validos en el archivo de matches.")

    return match_ids


def limpiar_carpeta_por_match_ids(folder: Path, match_ids: Set[int], aplicar: bool) -> tuple[int, int]:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"La carpeta no existe o no es valida: {folder}")

    total_json = 0
    eliminables = 0

    for json_file in folder.glob("*.json"):
        total_json += 1
        stem = json_file.stem
        try:
            file_match_id = int(stem)
        except ValueError:
            # Si el nombre no es numerico, no corresponde a un match_id.
            eliminables += 1
            if aplicar:
                json_file.unlink()
            continue

        if file_match_id not in match_ids:
            eliminables += 1
            if aplicar:
                json_file.unlink()

    return total_json, eliminables


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Limpia data/events y data/three-sixty para dejar solo los JSON "
            "cuyos nombres coincidan con los match_id de data/matches/9/281.json"
        )
    )
    parser.add_argument(
        "--matches-file",
        default="data/matches/9/281.json",
        help="Ruta al JSON de matches con los match_id permitidos.",
    )
    parser.add_argument(
        "--events-dir",
        default="data/events",
        help="Ruta a la carpeta events.",
    )
    parser.add_argument(
        "--three-sixty-dir",
        default="data/three-sixty",
        help="Ruta a la carpeta three-sixty.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Aplica el borrado real. Sin este flag, solo muestra lo que borraria.",
    )

    args = parser.parse_args()

    matches_file = Path(args.matches_file)
    events_dir = Path(args.events_dir)
    three_sixty_dir = Path(args.three_sixty_dir)

    if not matches_file.exists():
        raise FileNotFoundError(f"No existe el archivo de matches: {matches_file}")

    match_ids = cargar_match_ids(matches_file)

    print(f"Match IDs permitidos: {len(match_ids)}")
    print("Modo:", "APLICAR (borra archivos)" if args.apply else "SIMULACION (sin borrar)")

    total_events, borrables_events = limpiar_carpeta_por_match_ids(events_dir, match_ids, args.apply)
    total_360, borrables_360 = limpiar_carpeta_por_match_ids(three_sixty_dir, match_ids, args.apply)

    print("\nResumen")
    print(f"- events: {total_events} JSON, {borrables_events} {'eliminados' if args.apply else 'para eliminar'}")
    print(f"- three-sixty: {total_360} JSON, {borrables_360} {'eliminados' if args.apply else 'para eliminar'}")


if __name__ == "__main__":
    main()
