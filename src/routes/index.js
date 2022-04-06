import { Router } from 'express';

const indexRouter = Router();

/* GET home page. */
indexRouter.get('/', function(req, res, next) {
    res.render('index', { title: 'OU - Titanic ML' });
});

export { indexRouter };
