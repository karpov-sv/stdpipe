
import psycopg2, psycopg2.extras

import numpy as np

from astropy.table import Table


class DB:
    """
    Class encapsulating the connection to PostgreSQL database
    """

    def __init__(
        self,
        dbname=None,
        dbhost=None,
        dbport=None,
        dbuser=None,
        dbpassword=None,
        readonly=False,
    ):
        connstring = ""
        if dbname is not None:
            connstring += "dbname=" + dbname
        if dbhost is not None:
            connstring += " host=" + dbhost
        if dbport is not None:
            connstring += " port=%d" % dbport
        if dbuser is not None:
            connstring += " user=" + dbuser
        if dbpassword is not None:
            connstring += " password='%s'" % dbpassword

        self.connect(connstring, readonly)

    def connect(self, connstring, readonly=False):
        self.conn = psycopg2.connect(connstring)
        self.conn.autocommit = True
        self.conn.set_session(readonly=readonly)
        psycopg2.extras.register_default_jsonb(self.conn)
        # FIXME: the following adapter is registered globally!
        psycopg2.extensions.register_adapter(dict, psycopg2.extras.Json)
        psycopg2.extensions.register_adapter(np.float32, psycopg2.extensions.AsIs)
        psycopg2.extensions.register_adapter(np.float64, psycopg2.extensions.AsIs)

        self.connstring = connstring
        self.readonly = readonly

    def query(self, string="", data=(), table=True, simplify=True, verbose=False):
        log = (
            (verbose if callable(verbose) else print)
            if verbose
            else lambda *args, **kwargs: None
        )

        if self.conn.closed:
            log("DB connection is closed, re-connecting")
            self.connect(self.connstring, self.readonly)

        cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        if verbose:
            log('Sending DB query:', cur.mogrify(string, data))

        if data:
            cur.execute(string, data)
        else:
            cur.execute(string)

        try:
            result = cur.fetchall()

            if table:
                # Code from astrolibpy, https://code.google.com/p/astrolibpy
                strLength = 10
                __pgTypeHash = {
                    16: bool,
                    18: str,
                    20: 'i8',
                    21: 'i2',
                    23: 'i4',
                    25: '|S%d' % strLength,
                    700: 'f4',
                    701: 'f8',
                    1042: '|S%d' % strLength,  # character()
                    1043: '|S%d' % strLength,  # varchar
                    1114: '|O',  # datetime
                    1700: 'f8',  # numeric
                }

                desc = cur.description
                names = [d.name for d in desc]
                formats = [__pgTypeHash.get(d.type_code, '|O') for d in desc]

                # table = np.recarray(shape=(cur.rowcount,), formats=formats, names=names)
                table = np.recarray(shape=(cur.rowcount,), formats=formats, names=names)

                for i, v in enumerate(result):
                    table[i] = tuple(v)

                table = Table(table)

                return table

            elif simplify and len(result) == 1:
                # Simplify the result if it is simple
                if len(result[0]) == 1:
                    return result[0][0]
                else:
                    return result[0]
            else:
                return result
        except:
            # Nothing returned from the query
            # import traceback
            # traceback.print_exc()
            return None
