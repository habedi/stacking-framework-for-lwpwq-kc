COMPETITION=linking-writing-processes-to-writing-quality

kaggle competitions download -c $COMPETITION

rm -rf competition_data

unzip -d competition_data "$COMPETITION.zip"

rm "$COMPETITION.zip"