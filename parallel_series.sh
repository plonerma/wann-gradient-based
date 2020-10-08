#!/usr/bin/env bash

MAX_CHILDREN=8
SERIES_FILE=$1
STEP=1

if [ "$#" -ne 1 ]; then
    echo "Please provide some series spec file."
fi


. ../venv/bin/activate


NUM_EXPERIMENTS=$(python mnist_series.py --from $SERIES_FILE --num_experiments)

if [ $? -ne 0 ];
then
  ./rewann-remote/telegram-alert.sh "Something is wrong (GD; $SERIES_FILE; init)"
  exit 1
fi

function wait_for_child {
  wait -n

  if [ $? -ne 0 ];
  then
    ERROR_MSG=$(tail series.err)
    ./rewann-remote/telegram-alert.sh "Some experiment in $SERIES_FILE failed."
    exit 1
  fi
}

./rewann-remote/telegram-alert.sh "Starting series with $NUM_EXPERIMENTS experiments."



for START_AT in $(seq 0 $STEP $NUM_EXPERIMENTS); do
  STOP_AT=$((START_AT + $STEP))

  if [ $STOP_AT -ge $NUM_EXPERIMENTS ]; then
    STOP_AT=$NUM_EXPERIMENTS
  fi


  CHILDREN=$(pgrep -c -P$$)

  if [ $CHILDREN -ge $MAX_CHILDREN ]; then
    wait_for_child
  fi

  # execute experiments
  nice -19 python mnist_series.py --from $SERIES_FILE --set start_at=$START_AT stop_at=$STOP_AT &
  ./rewann-remote/telegram-alert.sh "Started experiments $SERIES_FILE [$START_AT : $STOP_AT] - pid: $!."
done

while [$(pgrep -c -P$$) -ge 0]; do
  wait_for_child
done

echo "Done."
