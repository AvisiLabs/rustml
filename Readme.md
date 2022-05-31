# This is a simple machine learning repo

To run this application install some version of rust 1.62-c nightly was used for this build.

Then use
``Cargo Run`` to run this application, it comes with pretrained models based on playlists mentioned in the spotify notebook.

These are saved in a Postgresql instance that should be defined and running. You might find that Postgresql or Python has made some assumption about your datatypes (from the
Spotify data).
I suggest you manually correct the datatypes to fit your Rust code in PgAdmin or sth like that.

For any further questions hit me up on Github.
