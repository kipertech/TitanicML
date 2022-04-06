import crypto from "crypto";
import moment from "moment";

// region Function - Get Total Execution Time
export function getExecutionTime(startedAt, unit = 'minutes')
{
    return moment.duration(moment().unix() - startedAt, 'seconds').as(unit).toFixed(1) + ' ' + unit;
}
// endregion

// region Function - This is Local environment
export function isLocal()
{
    return(process.env.IS_LOCAL?.toLowerCase() === 'true' || process.env.IS_LOCAL?.toLowerCase() === 'yes');
}
// endregion

// region Function - Find the date in string (MMMM YYYY)
export function getFirstDateFromString(text, excludeStr = '842', startingIndex = 0)
{
    let trimmedText = text.substring(startingIndex, text.length - 1).replace(new RegExp(excludeStr, 'g'), '');

    let allNumbers = trimmedText.match(/^\d+|\d+\b|\d+(?=\w)/g),
        firstYearOccurrence = allNumbers.find((num) => num.length === 4);

    if (firstYearOccurrence)
    {
        let firstYearIndex = trimmedText.indexOf(firstYearOccurrence),
            splitTextFromStartToFirstYear = trimmedText.substring(0, firstYearIndex - 1).split(' '),
            checkStr = splitTextFromStartToFirstYear[splitTextFromStartToFirstYear.length - 1];

        // Check if the "checkStr" is not actually the day
        let isDay = checkStr.indexOf(',') > -1,
            dateStr = '';

        if (isDay)
        {
            let dayStr = checkStr,
                monthStr = splitTextFromStartToFirstYear[splitTextFromStartToFirstYear.length - 2];

            dateStr = monthStr + ' ' + dayStr + ' ' + firstYearOccurrence;
        }
        else dateStr = checkStr + ' ' + firstYearOccurrence;

        return({ dateStr, isValid: moment(dateStr, isDay ? 'MMMM D, YYYY' : 'MMMM YYYY').isValid() });
    }
    else return({ dateStr: '(Not Found)', isValid: false });
}
// endregion

// region Function - Convert HTML to Plain Text
export function htmlToText(htmlStr)
{
    let html = htmlStr;
    html = html.replace(/<style([\s\S]*?)<\/style>/gi, '');
    html = html.replace(/<script([\s\S]*?)<\/script>/gi, '');
    html = html.replace(/<\/div>/ig, '\n');
    html = html.replace(/<\/li>/ig, '\n');
    html = html.replace(/<li>/ig, '  *  ');
    html = html.replace(/<\/ul>/ig, '\n');
    html = html.replace(/<\/p>/ig, '\n');
    html = html.replace(/<br\s*[\/]?>/gi, "\n");
    html = html.replace(/<[^>]+>/ig, '');

    return html;
}
// endregion

// region Get App URL
export function getAppURL(isPortal = false)
{
    return(`https://${isPortal ? 'portal' : 'app'}${process.env.IS_DEV === 'YES' ? '-dev' : ''}.pardiesforapurpose.com`);
}
// endregion

// region Format US Phone Number
export function formatPhoneNumber(phoneNumberString)
{
    if (typeof phoneNumberString === 'string' && phoneNumberString.trim().length > 0)
    {
        let cleaned = ('' + phoneNumberString.match(/\d/g)?.join("")).replace(/\D/g, '');

        if (!cleaned) return('');

        let match = cleaned.match(/^(1|)?(\d{3})(\d{3})(\d{4})$/);
        if (match)
        {
            let intlCode = (match[1] ? '+1 ' : '');
            return([intlCode, '(', match[2], ') ', match[3], '-', match[4]].join('').replace('+1', '').trim());
        }
        return phoneNumberString;
    }
    else return('');
}
// endregion

export function createMD5(text)
{
    return crypto.createHash('md5').update(text).digest("hex");
}
// endregion

// region Function - Capitalize First Character of all words in array
export function capitalizeFirstCharacterInArray(array = [], objKey = null)
{
    let newArr =
        array.map((item) =>
            (objKey ? item[objKey] : item)
                .split(' ')
                .map((s) => s.charAt(0).toUpperCase() + s.substring(1))
                .join(' ')
        );

    return(newArr);
}
// endregion

// region Function - Parse Error Object
export function parseErrorObject(error, showMessageOnly = true)
{
    if (typeof error === 'string')
    {
        return({ errorMessage: error });
    }
    else return(Object.assign({ errorMessage: error.message }, showMessageOnly ? {} : { errorData: error }));
}
// endregion

// region Function - Check if variable is object type
export function isObject(value)
{
    return(value && typeof value === 'object' && value.constructor === Object);
}
// endregion

// region Function - Check if string is valid JSON format
export function isValidJSON(str)
{
    if ( /^\s*$/.test(str) ) return false;
    str = str.replace(/\\(?:["\\\/bfnrt]|u[0-9a-fA-F]{4})/g, '@');
    str = str.replace(/"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g, ']');
    str = str.replace(/(?:^|:|,)(?:\s*\[)+/g, '');
    return(/^[\],:{}\s]*$/).test(str);
}
// endregion

// region Function - Get Time String
export function getTimeStr(unixTime)
{
    return(moment.unix(unixTime).tz('America/Chicago').format('MMM DD, YYYY - hh:mm A') + ` (CST)`);
}
// endregion
