## Reference: https://community.mis.temple.edu/zuyinzheng/pythonworkshop/
# Author: Shuning Miao

import os
import re
import csv
import time #"time" helps to break for the url visiting
from urllib.request import urlopen
from bs4 import BeautifulSoup

class parse_8K():
	"""Creates df_item.csv (right?)
	"""
	
    def __init__(self):
        self.FormType = "8-K"   ### <=== Type your document type here
        self.nbDocPause = 5 ### <=== Type your number of documents to download in one batch
        self.nbSecPause = 1

        self.htmlSubPath = "./HTML/" #<===The subfolder with the 8-K files in HTML format
        self.companyListFile = "CompanyList.csv" # a csv file with the list of company ticker symbols and names (the file has a line with headers)
        self.IndexLinksFile = "IndexLinks.csv"  #a csv file (output of the 1GetIndexLinks) with the list of index links for each firm (the file has a line with headers)
        self.Form8kListFile = "8kList.csv"    #a csv file (output of 2Get8kLink) with the list of 8-K links for each firm (the file has a line with headers)
        self.logFile = "8kDownloadLog.csv" #a csv file (output of 3DownloadHTML) with the download history of 8-K forms
        self.txtSubPath = "./txt/" #<===The subfolder with the extracted text files
        self.ReadLogFile = "8kReadlog.csv" #a csv file (output of the 4readHTML) showing whether item is successfully extracted from 8-K forms
        self.df_item = "df_item.csv"
    def getIndexLink(self,tickerCode,FormType):
        csvOutput = open(self.IndexLinksFile,"a+",newline='') # "a+" indicates that we are adding lines rather than replacing lines
        csvWriter = csv.writer(csvOutput, quoting = csv.QUOTE_NONNUMERIC)

        urlLink = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK="+tickerCode+"&type="+FormType+"&dateb=&owner=exclude&count=100"
        pageOpen = urlopen(urlLink)
        pageRead = pageOpen.read()

        soup = BeautifulSoup(pageRead,"html.parser")
        table = soup.find("table", { "class" : "tableFile2" })
        docIndex = 1
        #Check if there is a table to extract / code exists in edgar database
        try :
            for row in table.findAll("tr"):
                cells = row.findAll("td")
                if len(cells)==5:
                    if cells[0].text.strip() == FormType:
                        link = cells[1].find("a",{"id": "documentsbutton"})
                        docLink = "https://www.sec.gov"+link['href']
                        description = cells[2].text.encode('utf8').strip() #strip take care of the space in the beginning and the end
                        filingDate = cells[3].text.encode('utf8').strip()
                        #newfilingDate = filingDate.replace("-","_")  ### <=== Change date format from 2012-1-1 to 2012_1_1 so it can be used as part of 10-K file names
                        csvWriter.writerow([tickerCode, docIndex, docLink, description, filingDate])
                        docIndex = docIndex + 1
        except:
            print ("No tables found or no matching ticker symbol for ticker symbol for "+tickerCode)
            csvOutput.close()
            return -1

        csvOutput.close()

    def getIndexLink_all(self):
        csvFile = open(self.companyListFile,"r") #<===open and read from a csv file with the list of company ticker symbols (the file has a line with headers)
        csvReader = csv.reader(csvFile,delimiter=",")
        csvData = list(csvReader)
        csvOutput = open(self.IndexLinksFile,"a+",newline='') #<===open and write to a csv file which will include the list of index links. New rows will be appended.
        csvWriter = csv.writer(csvOutput, quoting = csv.QUOTE_NONNUMERIC)
        csvWriter.writerow(["Ticker", "DocIndex","IndexLink", "Description", "FilingDate"])
        csvOutput.close()
        i = 1
        for rowData in csvData[1:]:
            ticker = rowData[0]
            self.getIndexLink(ticker,self.FormType)
            if i%self.nbDocPause == 0:
                print(i)
                print("Pause for "+str(self.nbSecPause)+" second .... ")
                time.sleep(float(self.nbSecPause))
            i=i+1
        csvFile.close()

    def get8kLink(self,tickerCode, docIndex, docLink, description, filingDate, FormType):
        csvOutput = open(self.Form8kListFile, "a+",newline='')
        csvWriter = csv.writer(csvOutput, quoting = csv.QUOTE_NONNUMERIC)
        pageOpen = urlopen(docLink)
        pageRead = pageOpen.read()

        soup = BeautifulSoup(pageRead,"html.parser")

        #Check if there is a table to extract / code exists in edgar database
        try:
            table = soup.find("table", { "summary" : "Document Format Files" })
        except:
            print ("No tables found for link "+docLink)
            csvOutput.close()
            return -1

        for row in table.findAll("tr"):
            cells = row.findAll("td")
            if len(cells)==5:
                if cells[3].text.strip() == FormType:
                    link = cells[2].find("a")
                    formLink = "https://www.sec.gov"+link['href']
                    formName = link.text.encode('utf8').strip()
                    csvWriter.writerow([tickerCode, docIndex, docLink, description, filingDate, formLink,formName])
        csvOutput.close()

    def get8kLink_all(self):
        csvFile = open(self.IndexLinksFile,"r") #<===Open and read from a csv file with the list of index links for each firm (the file has a line with headers)
        csvReader = csv.reader(csvFile,delimiter=",")
        csvData = list(csvReader)

        csvOutput = open(self.Form8kListFile, "a+",newline='') #<===open and write to a csv file which will include the list of 8-K links. New rows will be appended.
        csvWriter = csv.writer(csvOutput, quoting = csv.QUOTE_NONNUMERIC)

        csvWriter.writerow(["Ticker", "DocIndex", "IndexLink", "Description", "FilingDate", "Form8KLink","Form8KName"])
        csvOutput.close()
        i = 1
        for rowData in csvData[1:]:
            Ticker = rowData[0]
            DocIndex = rowData[1]
            DocLink = rowData[2]
            Description = rowData[3]
            FileDate = rowData[4]
            self.get8kLink(Ticker,DocIndex,DocLink,Description,FileDate,self.FormType)
            if i%self.nbDocPause == 0:
                print (i)
                print ("Pause for "+str(self.nbSecPause)+" second .... ")
                time.sleep(float(self.nbSecPause))
            i=i+1

        csvFile.close()

    def downloadHTML(self,tickerCode, docIndex, docLink, description, filingDate, formLink,formName):
        csvOutput = open(self.logFile,"a+",newline='')
        csvWriter = csv.writer(csvOutput, quoting = csv.QUOTE_NONNUMERIC)

        try:
            pageOpen = urlopen(formLink)
            pageRead = pageOpen.read()
            htmlname = tickerCode+"_"+docIndex+"_"+filingDate+".htm"
            htmlpath = self.htmlSubPath+htmlname
            htmlfile = open(htmlpath,'wb')
            htmlfile.write(pageRead)
            htmlfile.close()
            csvWriter.writerow([tickerCode, docIndex, docLink, description, filingDate, formLink,formName, htmlname, ""])
        except:
            csvWriter.writerow([tickerCode, docIndex, docLink, description, filingDate, formLink,formName, "","not downloaded"])

        csvOutput.close()

    def downloadHTML_all(self):
        if not os.path.isdir(self.htmlSubPath):  ### <=== keep all HTML files in this subfolder
            os.makedirs(self.htmlSubPath)

        FormYears = ['2015','2016','2017','2018','2019'] ### <=== Type the years of documents to download here
        csvFile = open(self.Form8kListFile, "r",) #<===A csv file with the list of company ticker symbols (the file has a line with headers)
        csvReader = csv.reader(csvFile,delimiter=",")
        csvData = list(csvReader)

        csvOutput = open(self.logFile,"a+",newline='')
        csvWriter = csv.writer(csvOutput, quoting = csv.QUOTE_NONNUMERIC)

        csvWriter.writerow(["Ticker", "DocIndex", "IndexLink", "Description", "FilingDate", "Form8KLink","Form8KName", "FileName","Note"])
        csvOutput.close()

        i = 1
        for rowData in csvData[1:]:
            Ticker = rowData[0]
            DocIndex = rowData[1]
            IndexLink = rowData[2]
            Description = rowData[3]
            FilingDate = rowData[4]
            FormLink = rowData[5]
            FormName = rowData[6]
            for year in FormYears:
                if year in FilingDate:
                    if ".htm" in FormName:
                        self.downloadHTML(Ticker, DocIndex, IndexLink, Description, FilingDate, FormLink,FormName)
                    elif ".txt" in FormName:
                        csvOutput = open(self.logFile,"a+",newline='')
                        csvWriter = csv.writer(csvOutput, quoting = csv.QUOTE_NONNUMERIC)
                        csvWriter.writerow([Ticker, DocIndex, IndexLink, Description, FilingDate, FormLink,FormName, "","Text format"])
                        csvOutput.close()
                    else:
                        csvOutput = open(self.logFile,"a+",newline='')
                        csvWriter = csv.writer(csvOutput, quoting = csv.QUOTE_NONNUMERIC)
                        csvWriter.writerow([Ticker, DocIndex, IndexLink, Description, FilingDate, FormLink,FormName,"", "No form"])
                        csvOutput.close()

            if i%self.nbDocPause == 0:
                print (i)
                print ("Pause for "+str(self.nbSecPause)+" second .... ")
                time.sleep(float(self.nbSecPause))
            i=i+1

        csvFile.close()

    def readHTML(self,Ticker,FilingDate,file_name,dfWriter):
        input_path = self.htmlSubPath+file_name
        output_path = self.txtSubPath+file_name.replace(".htm",".txt")

        input_file = open(input_path,'r',encoding='utf-8')
        page = input_file.read()  #<===Read the HTML file into Python
        input_file.close()
        #Pre-processing the html content by removing extra white space and combining then into one line.
        page = page.strip()  #<=== remove white space at the beginning and end
        #page = page.replace('\n', ' ') #<===replace the \n (new line) character with space
        #page = page.replace('\r', '') #<===replace the \r (carriage returns -if you're on windows) with space
        page = page.replace('&nbsp;', ' ') #<===replace "&nbsp;" (a special character for space in HTML) with space.
        page = page.replace('&#160;', ' ') #<===replace "&#160;" (a special character for space in HTML) with space.
        while '  ' in page:
            page = page.replace('  ', ' ') #<===remove extra space
        #Using regular expression to extract texts that match a pattern

        #Define pattern for regular expression.
            #The following patterns find ITEM
            #(.+?) represents everything between the two subtitles
        #If you want to extract something else, here is what you should change
        '''
        First extract content between Item & SIGNATURE
        Sometimes fail (don't know why)
        Then find Item and then extract content between Item & SIGNATURE (never fail after testing)
        '''
        regexs = 'Item(.+?)SIGNATURE'
        #regexs_content = (';font-size:9pt;">(.+?)</font></div>',';font-size:10pt;">(.+?)</font></div>',)
        #Now we try to see if a match can be found...
        fail = False
        match = re.search(regexs, page, flags=re.IGNORECASE)  #<===search for the pattern in HTML using re.search from the re package. Ignore cases.
        #If a match exist....
        if match:
            #Now we have the extracted content still in an HTML format
            #We now turn it into a beautiful soup object
            #so that we can remove the html tags and only keep the texts
            output_file = open(output_path, "w",encoding='utf')

            soup = BeautifulSoup(match.group(1), "html.parser")
            rawText = soup.text
            output_file.write('Item'+rawText)
            dfWriter.writerow([Ticker, FilingDate, 'Item'+rawText])
            output_file.close()
            return match,fail
        else:
            regexs = 'Item(.+?)</font>'
            match = re.findall(regexs, page, flags=re.IGNORECASE)  #<===search for the pattern in HTML using re.search from the re package. Ignore cases.
            if match:
                output_file = open(output_path, "w",encoding='utf-8')
                for i in match:
                    soup = BeautifulSoup(i, "html.parser")
                    content = page.split(i,1)[1]
                    rawText = soup.text
            #remove space at the beginning
            #outText = re.sub("","",rawText.strip(),flags=re.IGNORECASE)
                    result = 'Item'+rawText+BeautifulSoup(content, "html.parser").text
                    try:
                        result = result.split('SIGNATURE')[0]
                    except:
                        print(Ticker,FilingDate,"Signature split failed")
                        fail = True
                    output_file.write(result)
                    dfWriter.writerow([Ticker, FilingDate, result])
                    output_file.close()
                    return match,fail
            else:
                return False,True

    def readHTML_all(self):
        if not os.path.isdir(self.txtSubPath):  ### <=== keep all texts files in this subfolder
            os.makedirs(self.txtSubPath)

        csvFile = open(self.logFile, "r") #<===A csv file with the list of 8k file names (the file should have no header)
        csvReader = csv.reader(csvFile,delimiter=",")
        csvData = list(csvReader)

        logFile = open(self.ReadLogFile, "a+",newline='') #<===A log file to track which file is successfully extracted
        logWriter = csv.writer(logFile, quoting = csv.QUOTE_NONNUMERIC)
        logWriter.writerow(["filename","extracted"])

        df = open(self.df_item, "a+",newline='',encoding='utf')
        dfWriter = csv.writer(df, quoting = csv.QUOTE_NONNUMERIC)
        dfWriter.writerow(["Ticker","Date","Content"])

        i=1
        for rowData in csvData[1:]:
           Ticker = rowData[0]
           # DocIndex = rowData[1]
           # IndexLink = rowData[2]
           # Description = rowData[3]
           FilingDate = rowData[4]
           # FormLink = rowData[5]
           # FormName = rowData[6]
           FileName = rowData[7]
           if ".htm" in FileName:
                match,fail_indicator=self.readHTML(Ticker,FilingDate,FileName,dfWriter)
                if match:
                    if not fail_indicator:
                        logWriter.writerow([FileName,"yes"])
                    else:
                        logWriter.writerow([FileName,"further look"])
                else:
                    logWriter.writerow([FileName,"no"])
           i=i+1

        csvFile.close()

        logFile.close()
        df.close()



    def run(self):
        ## First step : get all the IndexLink
        self.getIndexLink_all()
        print ("First step getIndexLink_all done!")
        ## Second step : get all the 8k Link
        self.get8kLink_all()
        print ("Second step get8kLink_all done!")
        ## Third step : download all the 8k HTML
        self.downloadHTML_all()
        print ("Third step downloadHTML_all done!")
        ## Fourth step:
        self.readHTML_all()
        print("Fourth step readHTML_all done!")


a = parse_8K()
a.run()
