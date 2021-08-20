const CronJob = require('cron').CronJob;
const mongoose = require('mongoose');

require('dotenv').config();

const connectionString = "mongodb+srv://" + process.env.db_user + ":" + process.env.db_pass + "@cluster0.tjbly.mongodb.net/" + process.env.db_name + "?retryWrites=true&w=majority";

mongoose.connect(connectionString, { useNewUrlParser: true, useUnifiedTopology: true });
const db = mongoose.connection;

const gameCollection = db.collection('games');
const userCollection = db.collection('users');

db.on('error', console.error.bind(console, 'connection error:'));
db.once('open', function () {
    console.log('Connected to Database');

    const job = new CronJob('* * * * * *', function() {
        // Delete unfinished games
        gameCollection.deleteMany({ 
            active: true,
            created: {"$lt": new Date(Date.now() - 5*60 * 1000) },
        }).then((res) => {
            console.log("Deleted", res.deletedCount, "unfinised games");
        }).catch(err => {
            console.error(err);
        });

        // Delete unactivated users
        userCollection.deleteMany({ 
            activated: false,
            created: {"$lt": new Date(Date.now() - 24*60*60 * 1000) },
        }).then((res) => {
            console.log("Deleted", res.deletedCount, "unactivated accounts");
        }).catch(err => {
            console.error(err);
        });
    }, null, true, 'Europe/London');
    

    job.start();
});





