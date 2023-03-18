import ROOT

def defaultStyle():
    st = ROOT.TStyle("defaultStyle", "mystyle")
    st.SetCanvasColor(ROOT.kWhite)
    st.SetCanvasBorderMode(0)
    st.SetFrameBorderMode(0)
    st.SetCanvasDefH(800)
    st.SetCanvasDefW(800)

    st.SetPadTickX(1)
    st.SetPadTickY(1)

    st.SetPadColor(ROOT.kWhite)

    # Margins:
    st.SetPadTopMargin(0.12)
    st.SetPadBottomMargin(0.12)
    st.SetPadLeftMargin(0.16)
    st.SetPadRightMargin(0.04)

    st.SetTitleFillColor(ROOT.kWhite)
    st.SetTitleBorderSize(0)

    st.SetTitleOffset(1.1, "x")
    st.SetTitleOffset(1.6, "y")

    st.SetStatBorderSize(1)
    st.SetStatColor(0)

    st.SetLegendBorderSize(0)
    st.SetLegendFillColor(ROOT.kWhite)
    st.SetLegendFont(st.GetLabelFont())
    # st.SetLegendTextSize( st.GetLabelSize() ) not in current ROOT version

    st.SetOptStat(0)

    textSize = 0.05
    st.SetLabelSize(textSize, "xyz")
    st.SetTitleSize(textSize, "xyz")

    st.SetTextFont(st.GetLabelFont())
    st.SetTextSize(st.GetLabelSize())

    st.SetNdivisions(505, "xyz")

    #st.SetPalette(56)
    st.SetNumberContours(999)

    # st.SetErrorX(0)
    # st.SetErrorX(0)

    st.cd()
    return st