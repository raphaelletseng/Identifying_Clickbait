import os
#requires pip install beautifulsoup4
from bs4 import BeautifulSoup
from bs4.element import Comment
#pip install newspaper3k
from newspaper import Article

#Just need to instantiate the Article, but we overwrite the html on each use
article = Article("http://bloom.bg/2gAMOiy")
article.download()

#This is the title as extracted by the Article class (the true headline)
def getArticleTitle(htmlOfArticle):
    article.html=htmlOfArticle
    article.parse()
    return article.title, article.text

#Twitter title from metadata:
def getTwitterTitle(soup):
    tag = soup.find(name="meta", attrs={"name":"twitter:title"})
    if tag == None:
        tag = soup.find(name="meta", attrs={"property":"twitter:title"})
    return tag['content'] if tag != None else ""


def tag_visible(element):
    #Ignore comments
    if isinstance(element, Comment):
        return False
    #Only look in the article tag
    if element.parent.find_parent(name='article') == None:
        return False
    #Ignore navigation content
    if element.find_parent(name='nav') != None:
        return False
    #Ignore asides
    if element.parent.find_parent(name='aside') != None:
        return False
    #Ignore header content
    if element.parent.find_parent(name='header') != None:
        return False

    
    
    #The only tags we consider for article body
    if element.parent.name not in ['p','a','li','h1','h2','h3','h4','h5','h6']:
        return False
    #Only consider link text if the link is in a paragraph, list, or for some reason a headline
    if element.parent.name == 'a' and element.parent.parent.name not in ['p','li','h1','h2','h3','h4','h5','h6']:
        return False
    
    return True

def getArticleText(soup):
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return (u" ".join(t.strip() for t in visible_texts)).strip()

def extractContent(htmlFile):
    soup = BeautifulSoup(htmlFile, 'html.parser')

    articleTitle, altBody= getArticleTitle(htmlFile)
    twitterTitle = getTwitterTitle(soup)
    articleBody = getArticleText(soup)
    
    return articleTitle, twitterTitle, articleBody, altBody

def readAndExtract(path, extractToFileName):
    stream = open(path, 'rb')
    html = stream.read()
    articleTitle, twitterTitle, articleBody, altBody = extractContent(html)
    stream.close()
    stream = open(extractToFileName, 'w')
    stream.write(articleTitle + '\n' + twitterTitle + '\n' + articleBody + '\n\n' + altBody)
    stream.close()

def main():
    initialDirectory = "archives-clickbait17-train"
    saveDir = "parsedHtml"
    
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    index = 1.00
    for subDir in [f.name for f in os.scandir(initialDirectory) if f.is_dir()]:
        print(index/100.0)
        index+=1
        for fileDir in [f.name for f in os.scandir(os.path.join(initialDirectory, subDir)) if f.is_dir()]:
            
            try:
                readAndExtract(os.path.join(initialDirectory, subDir, fileDir, "url_" + fileDir + ".html"), os.path.join(saveDir, fileDir + '.txt'))
            except:
                print(os.path.join(initialDirectory, subDir, fileDir, "url_" + fileDir + ".html"))
        
if __name__=="__main__":
        main()
