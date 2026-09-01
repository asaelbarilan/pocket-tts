# The tagging tool, packaged so several people can mark word boundaries in a browser.
#
# Deliberately not built on the training image: the server needs soundfile and numpy and
# nothing else, because the clips are already cut to their own wav files. That keeps this
# around 200 MB instead of the several GB a torch base would cost, which matters when the
# whole job is serving 35 MB of audio to a handful of annotators.
FROM python:3.12-slim

# soundfile is a binding, not an implementation -- without libsndfile it imports and then
# fails on the first read.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "soundfile==0.12.1" "numpy>=1.24"

WORKDIR /srv/app
COPY hebrew_training/align_tag_server.py hebrew_training/align_tag_page.html hebrew_training/
COPY data/gold_set/A_hebrew.jsonl data/gold_set/B_mms.jsonl data/gold_set/
COPY data/gold_set/clips/ data/gold_set/clips/
RUN touch hebrew_training/__init__.py

# Marks land here. Mount a volume on it: without one they live in the container's
# writable layer and a redeploy throws away everyone's hand-marking.
VOLUME /srv/marks
ENV HOST=0.0.0.0 PORT=8080
EXPOSE 8080

# --clips-root is what makes the manifests portable; the paths inside them are the Windows
# paths they were built with and resolve nowhere in here.
CMD ["python", "-m", "hebrew_training.align_tag_server", \
     "--a", "data/gold_set/A_hebrew.jsonl", \
     "--b", "data/gold_set/B_mms.jsonl", \
     "--clips-root", "data/gold_set/clips", \
     "--out", "/srv/marks", "--multi", "--limit", "100"]
