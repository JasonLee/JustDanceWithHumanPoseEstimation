import React, { useState, useEffect } from 'react'
import { useParams } from "react-router-dom";
import { Link } from 'react-router-dom'
import axios from 'axios';

// Email Confirmation when user clicks on verify email
export default function EmailConfirm() {
    const { id } = useParams();
    const [confirm, setConfirm] = useState(false);
    const [msg, setMsg] = useState("");

    // When the component mounts the mongo id for the user is pulled  from the 
    // params in React Router. This id is then sent to the server to confirm that 
    // the user has clicked on the link in the email. The link in the email will 
    // look something like this: 
    // 
    // http://localhost:3000/confirm/5c40d7607d259400989a9d42
    // 
    // where 5c40d...a9d42 is the unique id created by Mongo
    useEffect(() => {
        axios.post('/verifyAccount',{
            id: id
        }).then(res => {
            if(res) { 
                setConfirm(true)
                setMsg(res.data.msg);
            }
        }).catch(error => {
            console.error(error);
        })
    }, [id]);

    return (
        <div>
            {confirm && 
            <div style={{color:"white"}}>
                {msg} <br />
                <Link to='/'>
                    <button> Go to Home</button>
                </Link>
            </div>}
        </div>
    );
}