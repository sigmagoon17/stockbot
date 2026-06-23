from history_tracker import update_expired_history


def main():
    errors = update_expired_history(include_today=True)
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)

    print("Expiration history update completed.")


if __name__ == "__main__":
    main()
