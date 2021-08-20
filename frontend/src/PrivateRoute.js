import { Route, Redirect } from 'react-router-dom';
import ErrorBoundary from './components/ErrorBoundary';

export default function PrivateRoute({ component: Component, authed, token, ...rest }) {
    return (
        <Route
            {...rest}
            render={(props) => authed == true
                ? <ErrorBoundary> <Component token={token} /> </ErrorBoundary>
                : <Redirect to={{ pathname: '/login', state: { from: props.location } }} />}
        />
    )
}
