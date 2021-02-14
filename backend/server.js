const MongoClient = require('mongodb').MongoClient;
const express = require('express')
require('dotenv').config()

const app = express()
const port = 8000
const connectionString = "mongodb+srv://" + process.env.db_user + ":" + process.env.db_pass + "@cluster0.tjbly.mongodb.net/" + process.env.db_name + "?retryWrites=true&w=majority"

console.log(connectionString)
MongoClient.connect(connectionString, {
    useUnifiedTopology: true
}, (err, client) => {
    if (err) return console.error(err);
    console.log('Connected to Database');

    const db = client.db(process.env.dbname)
    const songCollection = db.collection('songs')
    
    app.get('/test', (req, res) => {
        songCollection.find().toArray()
        .then(results => {
            res.send(results);
        }).catch(error => console.error(error))
    })
});


app.listen(port, () => {
    console.log(`Server listening on port:${port}`);
});