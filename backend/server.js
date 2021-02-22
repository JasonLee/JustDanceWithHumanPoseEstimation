const mongoose = require('mongoose');
const express = require('express');
const ms = require('mediaserver');
require('dotenv').config();

const app = express();
const port = 8000;
const connectionString = "mongodb+srv://" + process.env.db_user + ":" + process.env.db_pass + "@cluster0.tjbly.mongodb.net/" + process.env.db_name + "?retryWrites=true&w=majority";

mongoose.connect(connectionString, {useNewUrlParser: true, useUnifiedTopology: true});
const db = mongoose.connection;

db.on('error', console.error.bind(console, 'connection error:'));
db.once('open', function() {
    console.log('Connected to Database');
});

const songCollection = db.collection('songs')

app.get('/songs', (req, res) => {
    songCollection.find().toArray()
    .then(results => {
        res.status(200).send(results);
    }).catch(error => console.error(error))
})

app.get('/audio', (req, res) => {
    ms.pipe(req, res, "../frontend/src/components/audio.mp3");
})

app.get('/audio2', (req, res) => {
    ms.pipe(req, res, "../frontend/src/components/audio2.mp3");
})

if (process.env.NODE_ENV != "test") {
    app.listen(port, () => {
        console.log(`Server listening on port:${port}`);
    });
}

module.exports = app;


