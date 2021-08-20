const redis = require("redis");
const client = redis.createClient();
const util = require('util');
client.get = util.promisify(client.get);

// Connects to Redis Server
client.on("connect", function(error) {
    console.log("Connected to Redis Server");
});

// Outputs Redis errors to stderr
client.on("error", function(error) {
    console.error(error);
});

// Join user to chat
async function userJoin(id, username, lobbyID) {
    const user = { "id": id, "username": username, "lobbyID": lobbyID };
    const data = await client.get(lobbyID);

    if(data === null) {
        strData = JSON.stringify([user]);
        client.set(lobbyID, strData);
    }else{
        usersD = JSON.parse(data)
        usersD.push(user);

        strData = JSON.stringify(usersD)

        client.set(lobbyID, strData);
    }

    strData = JSON.stringify(user);
    client.set(id, strData);

    return user;
}

// Get current user
async function getCurrentUser(id) {
    const data =  await client.get(id);
    return JSON.parse(data);
}

// User leaves chat
async function userLeave(id) {
    let user = await client.get(id);
    user = JSON.parse(user)

    if (!user) {
        return null;
    }

    lobbyID = user.lobbyID;

    let users = await client.get(lobbyID);
    users = JSON.parse(users);

    const index = users.findIndex(user => user.id === id);

    if (index !== -1) {
        users.splice(index, 1);
        usersStr = JSON.stringify(users);
        client.set(lobbyID, usersStr);
        
        console.log("USERS", user, id, users)
        client.del(id);
        return user;

    }

}

// Get room users
async function getRoomUsers(lobbyID) {
    const users = await client.get(lobbyID);
    return JSON.parse(users);
}

module.exports = {
    userJoin,
    getCurrentUser,
    userLeave,
    getRoomUsers,
};
