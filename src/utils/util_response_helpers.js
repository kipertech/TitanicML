export function Success(res, message, code = 200, data, pagination = {})
{
    res.status(code).send(Object.assign({ message, data }, pagination));
}

export function Abort(res, message, code = 500, errorData = null, extraData = null)
{
    res.status(code).send({
        message,
        error: errorData,
        extraData
    });
}
