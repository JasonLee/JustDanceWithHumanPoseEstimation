import React from 'react';
import styles from './css/ErrorPage.module.css';
import { Link } from 'react-router-dom';

// Error page redirection
export default class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false };
        this.logErrorToMyService = console.error;
    }

    static getDerivedStateFromError(error) {
        // Update state so the next render will show the fallback UI.
        return { hasError: true };
    }

    componentDidCatch(error, errorInfo) {
        this.logErrorToMyService(error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            // You can render any custom fallback UI
            return (
                <div className={styles.Mainox}>
                    <div className={styles.Head}>ERROR</div>
                    <div className={styles.Message}>
                        
                        Something went wrong. <br /><br />

                        Click here to go to the <Link to="/login" replace> Home Page</Link>
                    </div>
                </div>

            );
        }

        return this.props.children;
    }
}