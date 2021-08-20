import React from 'react';
import { Link } from 'react-router-dom';
import styles from './css/StartPage.module.css';
import Particles from 'react-tsparticles';
import { Button } from '@material-ui/core/';

// Custom particle parameters
const params = {
    particles: {
        number: {
            value: 100,
            density: {
                enable: true,
                value_area: 800
            }
        },
        color: {
            value: ["#c311e7", "#b8e986", "#4dc9ff", "#ffd300", "#ff7e79", "BC2C1A", "F17300"]
        },
        shape: {
            type: "circle",
            polygon: {
                nb_sides: 5
            }
        },
        opacity: {
            value: 0.9,
            random: false,
            anim: {
                enable: false,
                speed: 1,
                opacity_min: 0.5,
                sync: false
            }
        },
        size: {
            value: 8,
            random: true,
            anim: {
                enable: false,
                speed: 30,
                size_min: 0.1,
                sync: false
            }
        },
        move: {
            enable: true,
            speed: 3,
            direction: "none",
            random: false,
            straight: false,
            out_mode: "bounce",
            bounce: false,
            attract: {
                enable: false,
                rotateX: 600,
                rotateY: 1200
            }
        }
    },
    retina_detect: true
};

// Welcome Start Page
export default function StartPage() {
    return (
        <div>
            <div className={styles.wrapper}>
                <div className={styles.title} > Just Dance with Human Pose Estimation </div>
                <br />
                <Link to="/songs">
                    <Button  variant="contained" color="primary"> Start </Button>
                </Link>
            </div>
            <Particles className={styles.particles} width="100vw" height="99vh" params={params} />
        </div>
    )
}
