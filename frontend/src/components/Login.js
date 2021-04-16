import React, { useState } from 'react';
import PropTypes from 'prop-types';
import axios from 'axios';
import { Redirect } from 'react-router-dom';
import styles from './css/Login.module.css';
import avatar from '../images/img_avatar.png';

export default function Login({ setToken }) {
    const [username, setUserName] = useState();
    const [password, setPassword] = useState();
    const [redirect, setRedirect] = useState();

    const handleSubmit = async e => {
        e.preventDefault();

        axios.post("/login", {
            username: username,
            password: password
        }).then(res => {
            const token = res.data;
            setToken(token);
            setRedirect(true);
        }).catch(error => {
            console.log(error);
        });
    }

    if (redirect) {
        return <Redirect to='/songs'/>;
    }

    return (
        <div className={styles.LoginWrapper}>
            <h1 className={styles.Container} >Please Log In</h1>
            <form onSubmit={handleSubmit}>
                <div className={styles.ImgContainer}>
                    <img className={styles.avatar} src={avatar}/>
                </div>
                <div className={styles.Container}> 
                    <label>
                        <label><b>Username</b></label>
                        <input className={styles.input} type="text" onChange={e => setUserName(e.target.value)} required="required" />
                    </label>
                    <label>
                        <label><b>Password</b></label>
                        <input className={styles.input} type="password" onChange={e => setPassword(e.target.value)} required="required" />
                    </label>
                </div>
                <div className={styles.Container}>
                    <button className={styles.SubmitButton} type="submit">Login</button>
                </div>
            </form>
        </div>
    )
}

Login.propTypes = {
    setToken: PropTypes.func.isRequired
};
