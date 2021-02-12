const express = require('express')
const app = express()
const port = 8000

app.get('/test', (req, res) => {
    res.send('Hello World!')
})

app.listen(port, () => {
    console.log(`Server listening on port:${port}`);
});