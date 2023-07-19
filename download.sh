#!/bin/bash

while read -p "Choose a resource to download [wiki]: " choice; do
    case "$choice" in
        wiki )
            TARGET=$choice
            TARGET_DIR=$DATA_DIR
            break ;;
        * ) echo "Please type among [wiki]";
            exit 0 ;;
    esac
done

echo "$TARGET will be downloaded at $TARGET_DIR"

# Download + untar + rm
case "$TARGET" in
    wiki )
        wget -O "$TARGET_DIR/wikidump.tar.gz" "https://nlp.cs.princeton.edu/projects/densephrases/wikidump.tar.gz"
        tar -xzvf "$TARGET_DIR/wikidump.tar.gz" -C "$TARGET_DIR"
        rm "$TARGET_DIR/wikidump.tar.gz" ;;
    * ) echo "Wrong target $TARGET";
        exit 0 ;;
esac

echo "Downloading $TARGET done!"
