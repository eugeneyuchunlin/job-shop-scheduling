/**
 * @file csv_t.h
 * @brief The definition of csv_t object
 *
 * csv_t type is defined in this file. csv_t object is used to read and write
 * csv_t file.
 *
 * @author Eugene Lin <lin.eugene.l.e@gmail.com>
 * @date 2021.5.5
 */
#ifndef __CSV_H__
#define __CSV_H__

#include <bits/stdint-uintn.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <private/common.h>

/**
 * @class csv_t
 * @breif csv_t type is used to read/write csv_t file.
 *
 * csv_t object is used to deel with csv_t file I/O. There are several csv_t
 * file have byte order remark. csv_t object is able to handle UTF-8 and UTF-16
 * byte order remark.
 */
class csv_t
{
protected:
    /**
     * _hasBOM () - check if @b _text has UTF-@b bits kind of byte order mark.
     *
     * In _hasBOM, the first 4 characters are cast to be unsigned int type
     * variable and check if the variable has BOM.
     *
     * @param _text : first line in the file.
     * @param bom : byte order mark number.
     * @param bits : specify what kind of UTF
     *
     * @warning :
     * 1. Make sure @b _text has at least 4 bytes. If @b _text doesn't, the
     * behavior is undefined.
     * 2. The function only supports little endian hardward.
     */
    bool _hasBOM(char *_text, unsigned int bom, short bits);

protected:
    std::vector<std::vector<std::string> > _data;
    std::map<std::string, std::uint16_t> _head;
    std::string _mode;
    std::string _filename;
    FILE *_file;

    /**
     * parseCsvRow () - parse a row in csv file
     *
     * This function is able to parse a row, which contains special characters,
     * such as '"' and ',', in an element in row. The special character '"' in
     * an element follows another '"', and the element is quoted. For example,
     * the representation of an element with special character the sheet  is
     * A"B. The representation in the csv file is "A""B". For another example,
     * the representation of an element in the sheet is A,B. The representation
     * in csv file is "A,B". parseCsvRow is able to parse these kind of elements
     * to recover the original representation.
     *
     * @param text : pointer to character type of variable, which points on a
     * row in csv file
     * @param delimiter : delimiter in csv file
     */
    std::vector<std::string> parseCsvRow(char *text, char delimiter);

    std::string formCsvElement(std::string text);

public:
    csv_t();

    csv_t(csv_t &csv);

    /**
     * csv_t - Constructor of csv_t object for reading/writing file
     */
    csv_t(std::string filename, std::string mode);

    /**
     * csv_t - Constructor of csv_t object for reading file
     *
     * This constructor is used to read csv_t file. The mode of opening file is
     * specified in @b mode. if @b read is @b true the constructor will
     * immediately read the data. @b head is used to specify the csv_t file has
     * header, if head is set true but the head has empty columns, the
     * constructor raise exception. @b r1 and @b r2 are used to specify the
     * range of rows of csv_t file. If both are -1, all of data are read into to
     * memory. If @b r1 isn't -1 but @b r2 is, rows from r1 to the end are read
     * into memory.
     *
     * @var filename : the csv_t file name
     * @var mode : the mode of opening file
     * @var read : read data immediately or not
     * @var head : specify if data has header
     * @var r1 : range from r1 if specify
     * @var r2 : range to r2 if specify
     */
    csv_t(std::string filename,
          std::string mode,
          bool read,
          bool head,
          int r1 = -1,
          int r2 = -1);

    void trim(std::string text);

    /**
     * setMode () - set the file mode
     *
     * @var mode : file mode
     */
    void setMode(std::string mode);

    /**
     * setFileName() - set the csv_t file path
     *
     * @var filename : csv_t file path
     */
    void setFileName(std::string filename);

    /**
     * setHeaders() - set header by mapping new header to old header
     *
     * This function is used to change the header by using mapping. The
     * parameter @b maps store the relation ship between new header and old
     * header, i.e. new_header --map--> old_header. Parameter @b replace specify
     * if the function add new header or replace old header by new header.
     *
     * @var maps : a container store the relationship between new header and old
     * header.
     * @var replace : if the function add new header or replace the old header.
     */
    void setHeaders(std::map<std::string, std::string> maps,
                    bool replace = false);

    /**
     * setHeaders() - directly set header
     *
     * @var head : new header
     */
    void setHeaders(std::map<std::string, uint16_t> head);

    /**
     * setHeader() - set single header
     *
     * This function is used to add @b new_header or replace @b old_header by @b
     * new_header. @b replace is used to specify if the function perform
     * addition or replacment.
     *
     * @var old_header : specify which of index of this column name is going to
     * be mapped by new_header
     * @var new_header : new header name
     * @var replace : specify if function function perform addition or
     * replacement.
     */
    void setHeader(std::string old_header,
                   std::string new_header,
                   bool replace = false);

    /**
     * write() - output csv_t file
     *
     * write is used to write out the dataframe to a csv file.
     *
     * @param filename : specify the output filename. The default filename is
     * csv_t::filename
     * @param mode : specify the output mode. The default mode is csv_t::mode
     * @param head : specify if the output csv file need to output header or
     * just data
     */
    bool write(std::string filename = "",
               std::string mode = "",
               bool head = true);

    /**
     * read() - input csv_t file
     *
     * This function is used to read csv_t file. The mode of opening file is
     * specified in @b mode. if @b read is @b true the constructor will
     * immediately read the data. @b head is used to specify the csv_t file has
     * header, if head is set true but the head has empty columns, the
     * constructor raise exception. @b r1 and @b r2 are used to specify the
     * range of rows of csv_t file. If both are -1, all of data are read into to
     * memory. If @b r1 isn't -1 but @b r2 is, rows from r1 to the end are read
     * into memory.
     *
     * if filename or mode isn't specified, the function use the data member
     * filename or mode.
     *
     * @return true if read file successfully
     *
     * @var filename : csv_t file path
     * @var mode : file opening mode
     * @var head : specify if this csv_t file has header
     * @var r1 : lower bound
     * @var r2 : upper bound
     */
    bool read(std::string filename = "",
              std::string mode = "",
              bool head = true,
              int r1 = -1,
              int r2 = -1);

    /**
     * read() - read csv_t file use data member filename and mode
     *
     * @b head is used to specify the csv_t file has
     * header, if head is set true but the head has empty columns, the
     * constructor raise exception. @b r1 and @b r2 are used to specify the
     * range of rows of csv_t file. If both are -1, all of data are read into to
     * memory. If @b r1 isn't -1 but @b r2 is, rows from r1 to the end are read
     * into memory.
     *
     * @return true if read file successfully
     *
     * @var head : specify if this csv_t file has header
     * @var r1 : lower bound
     * @var r2 : upper bound
     */
    bool read(bool head = true, int r1 = -1, int r2 = -1);

    /**
     * close() - reset file pointer
     */
    void close();

    /**
     * nrows() - return number of rows
     *
     * @return number of rows
     */
    unsigned int nrows();


    /**
     * getRow() - return a row
     *
     * return @b row of the csv_t file if row exceed the number of rows of
     * csv_t, the function will throw exception. @b row can be less than 0, the
     * function will count the row from the back and return it.
     *
     * @return a vector of string
     */
    std::vector<std::string> getRow(int row);

    /**
     * getElement() - get single string in the csv_t file
     *
     * this function is used to get single string of the csv_t file by specifing
     * which row(number) and which col(number). @b row can be less than 0, the
     * function will count the row from the back and return it.
     *
     *
     * @return single string in the csv_t file
     *
     */
    std::string getElement(int col, int row);

    /**
     * getElement() - get single string in the csv_t file
     *
     * this function is used to get single string of the csv_t file by specifing
     * which row(number) and which col(number). @b row can be less than 0, the
     * function will count the row from the back and return it.
     *
     *
     * @return single string in the csv_t file
     *
     */
    std::string getElement(std::string col, int row);

    /**
     * getElements() - get a row in the csv_t file
     *
     * this function is used to return row content its header. The return type
     * is map<string, string> the mapping relation is header -> row element
     *
     * @return map<string, string> data which is relationship between header and
     * row elements
     *
     */
    std::map<std::string, std::string> getElements(int nrow);

    /**
     * getData() - get the csv_t content
     *
     * @return csv_t data as 2-D array which is represented in
     * vector<vector<string> >
     */
    std::vector<std::vector<std::string> > getData(int r1 = -1, int r2 = -1);

    /**
     * getColumn() - get column by specifing header
     *
     * @var header : spcified which column to return
     * @return 1-D string array
     */
    std::vector<std::string> getColumn(std::string header);


    /**
     * filter() - get a new csv_t object by specifing that column's name == value
     */
    csv_t filter(std::string head, std::string value);

    /**
     * addData() - add a row to dataframe
     *
     * addData is a function used to add data to dataframe. The type of
     * parameter is std::map<std::string, std::string> which means that the
     * parameter is a mapping relationship between headers and elements. For
     * example : elements["HEADER"] == DATA.
     *
     * @param elements : a mapping relationship between headers and data
     */
    void addData(std::map<std::string, std::string> elements);

    /**
     * dropNullRow() - drop the empty row
     *
     * After reading the csv file, there maybe lots of empty row in the
     * dataframe. dropNullRow will drop all empty row of the dataframe.
     */
    void dropNullRow();

    ~csv_t();
};

#endif
